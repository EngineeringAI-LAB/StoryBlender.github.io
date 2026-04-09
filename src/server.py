# blender_mcp_server.py
from mcp.server.fastmcp import FastMCP, Context, Image
import socket
import json
import asyncio
import logging
import tempfile
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List, Optional
import os
from pathlib import Path
import base64
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BlenderMCPServer")

# Default configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9876

@dataclass
class BlenderConnection:
    host: str
    port: int
    sock: socket.socket = None  # Changed from 'socket' to 'sock' to avoid naming conflict
    
    def connect(self) -> bool:
        """Connect to the Blender addon socket server"""
        if self.sock:
            return True
            
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to Blender at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Blender: {str(e)}")
            self.sock = None
            return False
    
    def disconnect(self):
        """Disconnect from the Blender addon"""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting from Blender: {str(e)}")
            finally:
                self.sock = None

    def receive_full_response(self, sock, buffer_size=8192):
        """Receive the complete response, potentially in multiple chunks"""
        chunks = []
        # Use a consistent timeout value that matches the addon's timeout
        sock.settimeout(3600.0)  # Match the addon's timeout
        
        try:
            while True:
                try:
                    chunk = sock.recv(buffer_size)
                    if not chunk:
                        # If we get an empty chunk, the connection might be closed
                        if not chunks:  # If we haven't received anything yet, this is an error
                            raise Exception("Connection closed before receiving any data")
                        break
                    
                    chunks.append(chunk)
                    
                    # Check if we've received a complete JSON object
                    try:
                        data = b''.join(chunks)
                        json.loads(data.decode('utf-8'))
                        # If we get here, it parsed successfully
                        logger.info(f"Received complete response ({len(data)} bytes)")
                        return data
                    except json.JSONDecodeError:
                        # Incomplete JSON, continue receiving
                        continue
                except socket.timeout:
                    # If we hit a timeout during receiving, break the loop and try to use what we have
                    logger.warning("Socket timeout during chunked receive")
                    break
                except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                    logger.error(f"Socket connection error during receive: {str(e)}")
                    raise  # Re-raise to be handled by the caller
        except socket.timeout:
            logger.warning("Socket timeout during chunked receive")
        except Exception as e:
            logger.error(f"Error during receive: {str(e)}")
            raise
            
        # If we get here, we either timed out or broke out of the loop
        # Try to use what we have
        if chunks:
            data = b''.join(chunks)
            logger.info(f"Returning data after receive completion ({len(data)} bytes)")
            try:
                # Try to parse what we have
                json.loads(data.decode('utf-8'))
                return data
            except json.JSONDecodeError:
                # If we can't parse it, it's incomplete
                raise Exception("Incomplete JSON response received")
        else:
            raise Exception("No data received")

    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a command to Blender and return the response"""
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Blender")
        
        command = {
            "type": command_type,
            "params": params or {}
        }
        
        try:
            # Log the command being sent
            logger.info(f"Sending command: {command_type} with params: {params}")
            
            # Send the command
            self.sock.sendall(json.dumps(command).encode('utf-8'))
            logger.info(f"Command sent, waiting for response...")
            
            # Set a timeout for receiving - use the same timeout as in receive_full_response
            self.sock.settimeout(15.0)  # Match the addon's timeout
            
            # Receive the response using the improved receive_full_response method
            response_data = self.receive_full_response(self.sock)
            logger.info(f"Received {len(response_data)} bytes of data")
            
            response = json.loads(response_data.decode('utf-8'))
            logger.info(f"Response parsed, status: {response.get('status', 'unknown')}")
            
            if response.get("status") == "error":
                logger.error(f"Blender error: {response.get('message')}")
                raise Exception(response.get("message", "Unknown error from Blender"))
            
            return response.get("result", {})
        except socket.timeout:
            logger.error("Socket timeout while waiting for response from Blender")
            # Don't try to reconnect here - let the get_blender_connection handle reconnection
            # Just invalidate the current socket so it will be recreated next time
            self.sock = None
            raise Exception("Timeout waiting for Blender response - try simplifying your request")
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            logger.error(f"Socket connection error: {str(e)}")
            self.sock = None
            raise Exception(f"Connection to Blender lost: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Blender: {str(e)}")
            # Try to log what was received
            if 'response_data' in locals() and response_data:
                logger.error(f"Raw response (first 200 bytes): {response_data[:200]}")
            raise Exception(f"Invalid response from Blender: {str(e)}")
        except Exception as e:
            logger.error(f"Error communicating with Blender: {str(e)}")
            # Don't try to reconnect here - let the get_blender_connection handle reconnection
            self.sock = None
            raise Exception(f"Communication error with Blender: {str(e)}")

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle"""
    # We don't need to create a connection here since we're using the global connection
    # for resources and tools
    
    try:
        # Just log that we're starting up
        logger.info("BlenderMCP server starting up")
        
        # Try to connect to Blender on startup to verify it's available
        try:
            # This will initialize the global connection if needed
            blender = get_blender_connection()
            logger.info("Successfully connected to Blender on startup")
        except Exception as e:
            logger.warning(f"Could not connect to Blender on startup: {str(e)}")
            logger.warning("Make sure the Blender addon is running before using Blender resources or tools")
        
        # Return an empty context - we're using the global connection
        yield {}
    finally:
        # Clean up the global connection on shutdown
        global _blender_connection
        if _blender_connection:
            logger.info("Disconnecting from Blender on shutdown")
            _blender_connection.disconnect()
            _blender_connection = None
        logger.info("BlenderMCP server shut down")

# Create the MCP server with lifespan support
mcp = FastMCP(
    "BlenderMCP",
    lifespan=server_lifespan
)

# Resource endpoints

# Global connection for resources (since resources can't access context)
_blender_connection = None
_polyhaven_enabled = True  # Add this global variable

def get_blender_connection():
    """Get or create a persistent Blender connection"""
    global _blender_connection, _polyhaven_enabled  # Add _polyhaven_enabled to globals
    
    # If we have an existing connection, check if it's still valid
    if _blender_connection is not None:
        try:
            # First check if Blender is still running by sending a ping command
            result = _blender_connection.send_command("get_scene_info")
            return _blender_connection
        except Exception as e:
            # Connection is dead, close it and create a new one
            logger.warning(f"Existing connection is no longer valid: {str(e)}")
            try:
                _blender_connection.disconnect()
            except:
                pass
            _blender_connection = None
    
    # Create a new connection if needed
    if _blender_connection is None:
        host = os.getenv("BLENDER_HOST", DEFAULT_HOST)
        port = int(os.getenv("BLENDER_PORT", DEFAULT_PORT))
        _blender_connection = BlenderConnection(host=host, port=port)
        if not _blender_connection.connect():
            logger.error("Failed to connect to Blender")
            _blender_connection = None
            raise Exception("Could not connect to Blender. Make sure the Blender addon is running.")
        logger.info("Created new persistent connection to Blender")
    
    return _blender_connection


@mcp.tool()
def get_scene_info(ctx: Context) -> str:
    """Get detailed information about the current Blender scene"""
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_scene_info")
        
        # Just return the JSON representation of what Blender sent us
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting scene info from Blender: {str(e)}")
        return f"Error getting scene info: {str(e)}"

@mcp.tool()
def get_object_info(ctx: Context, object_name: str) -> str:
    """
    Get detailed information about a specific object in the Blender scene.
    
    Parameters:
    - object_name: The name of the object to get information about
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_object_info", {"name": object_name})
        
        # Just return the JSON representation of what Blender sent us
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting object info from Blender: {str(e)}")
        return f"Error getting object info: {str(e)}"

@mcp.tool()
def get_viewport_screenshot(ctx: Context, max_size: int = 800, format: str = "png", selected_object_outline: bool = True, shading: str = "material") -> Image:
    """
    Capture a screenshot of the current Blender 3D viewport.
    
    Parameters:
    - max_size: Maximum size in pixels for the largest dimension (default: 800)
    - format: Image format (png, jpg, etc.) (default: "png")
    - selected_object_outline: Whether to show selection outlines (default: True)
    - shading: Viewport shading mode ('material' or 'rendered', default: 'material')
    
    Returns the screenshot as an Image.
    """
    try:
        blender = get_blender_connection()
        
        # Create temp file path
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"blender_screenshot_{os.getpid()}.png")
        
        result = blender.send_command("get_viewport_screenshot", {
            "max_size": max_size,
            "filepath": temp_path,
            "format": format,
            "selected_object_outline": selected_object_outline,
            "shading": shading
        })
        
        if "error" in result:
            raise Exception(result["error"])
        
        if not os.path.exists(temp_path):
            raise Exception("Screenshot file was not created")
        
        # Read the file
        with open(temp_path, 'rb') as f:
            image_bytes = f.read()
        
        # Delete the temp file
        os.remove(temp_path)
        
        return Image(data=image_bytes, format="png")
        
    except Exception as e:
        logger.error(f"Error capturing screenshot: {str(e)}")
        raise Exception(f"Screenshot failed: {str(e)}")






@mcp.tool()
def get_object_image(ctx: Context, direction: str = "front", view_distance: str = "close", max_size: int = 800, format: str = "png", selected_object_outline: bool = True, shading: str = "material") -> Image:
    """
    Align the 3D viewport to show the specified direction of the currently active object,
    then frame it, adjust the zoom distance, and take a screenshot.
    
    Parameters:
    - direction: The direction to align to. Options: 'front', 'back', 'top', 'bottom', 'left', 'right' (default: 'front')
    - view_distance: The zoom distance after framing. Options: 'close', 'medium', 'far' (default: 'close')
                    - 'close': no zoom out
                    - 'medium': zoom out 3 steps  
                    - 'far': zoom out 6 steps
    - max_size: Maximum size in pixels for the largest dimension of the image (default: 800)
    - format: Image format (png, jpg, etc.) (default: "png")
    - selected_object_outline: Whether to show selection outlines (default: True)
    - shading: Viewport shading mode ('material' or 'rendered', default: 'material')
    
    Returns the screenshot as an Image.
    """
    try:
        blender = get_blender_connection()
        
        # Create temp file path
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"blender_object_image_{os.getpid()}.png")
        
        result = blender.send_command("get_object_image", {
            "direction": direction,
            "view_distance": view_distance,
            "max_size": max_size,
            "filepath": temp_path,
            "format": format,
            "selected_object_outline": selected_object_outline,
            "shading": shading
        })

        if isinstance(result, dict) and result.get("error"):
            raise Exception(result["error"])

        if not os.path.exists(temp_path):
            raise Exception("Object image file was not created")
        
        # Read the file
        with open(temp_path, 'rb') as f:
            image_bytes = f.read()
        
        # Delete the temp file
        os.remove(temp_path)
        
        return Image(data=image_bytes, format="png")
        
    except Exception as e:
        logger.error(f"Error capturing object image: {str(e)}")
        raise Exception(f"Object image capture failed: {str(e)}")


@mcp.tool()
def execute_blender_code(ctx: Context, code: str) -> str:
    """
    Execute arbitrary Python code in Blender. Make sure to do it step-by-step by breaking it into smaller chunks.
    
    Parameters:
    - code: The Python code to execute
    """
    try:
        # Get the global connection
        blender = get_blender_connection()
        result = blender.send_command("execute_code", {"code": code})
        return f"Code executed successfully: {result.get('result', '')}"
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        return f"Error executing code: {str(e)}"


@mcp.tool()
def lighting_designer(ctx: Context, scene_description: Optional[str] = None, asset_id: Optional[str] = None, categories_limitation: Optional[List[str]] = None) -> str:
    """
    Design lighting for a scene using AI-selected HDRIs from Polyhaven.
    This tool uses semantic search with pre-computed embeddings for efficient HDRI matching,
    with LLM-based reranking for improved relevance.
    
    Parameters:
    - scene_description: Description of the scene lighting needs (e.g., "afternoon woods", "urban sunset", "studio lighting").
                        Ignored if asset_id is provided.
    - asset_id: Optional Polyhaven HDRI asset ID to use directly, bypassing AI selection.
    - categories_limitation: Optional list of category strings. If provided, only HDRIs containing ALL
                            specified categories will be returned (e.g., ["sunrise-sunset", "pure skies"]).
    
    Returns a message indicating success or failure with details about the applied lighting.
    """
    try:
        blender = get_blender_connection()
        
        params = {}
        if asset_id:
            params["asset_id"] = asset_id
        elif scene_description:
            params["scene_description"] = scene_description
        else:
            return "Error: Either scene_description or asset_id must be provided"
        
        if categories_limitation:
            params["categories_limitation"] = categories_limitation
        
        result = blender.send_command("lighting_designer", params)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if result.get("success"):
            message = result.get("message", "lighting design completed successfully")
            asset_id = result.get("asset_id", "")
            hdri_name = result.get("hdri_name", "")
            combined_score = result.get("combined_score")
            llm_score = result.get("llm_score")
            
            output = f"{message}\n\n"
            output += f"Selected HDRI: {asset_id}"
            if hdri_name:
                output += f" ({hdri_name})"
            output += "\n"
            if combined_score is not None:
                output += f"Semantic match score: {combined_score:.4f}\n"
            if llm_score is not None:
                output += f"LLM relevance score: {llm_score}/10\n"
            output += f"The HDRI has been applied as the world environment lighting."
            
            return output
        else:
            return f"Failed to design lighting: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error in lighting designer: {str(e)}")
        return f"Error in lighting designer: {str(e)}"


@mcp.tool()
def environment_artist(
    ctx: Context,
    ground_description: str = None,
    asset_id: str = None,
    categories_limitation: List[str] = None,
    width: float = 10.0,
    depth: float = 10.0,
    wall_description: str = None,
    wall_asset_id: str = None,
    wall_categories_limitation: List[str] = None,
    wall_x: float = None,
    wall_x_negative: float = None,
    wall_y: float = None,
    wall_y_negative: float = None,
    wall_z: float = None
) -> str:
    """
    Create a ground plane and optionally indoor walls using textures from Polyhaven.
    
    Uses semantic search with pre-computed embeddings for efficient texture matching,
    with optional LLM-based reranking for improved relevance.
    
    Parameters:
    - ground_description: Description of the ground/terrain (e.g., "cobblestone floor", "grassy meadow").
                          Ignored if asset_id is provided.
    - asset_id: Optional Polyhaven texture asset ID to use directly, bypassing AI selection.
    - categories_limitation: Optional list of category strings. If provided, only textures
                            containing ALL specified categories will be returned (e.g., ["floor", "cobblestone"]).
    - width: Width of the ground plane in meters (default: 10.0)
    - depth: Depth of the ground plane in meters (default: 10.0)
    - wall_description: Description of wall texture for indoor scenes. Ignored if wall_asset_id is provided.
    - wall_asset_id: Optional Polyhaven texture asset ID for walls, bypassing AI selection.
    - wall_categories_limitation: Optional category filter for wall textures.
    - wall_x: Positive X boundary for walls (optional).
    - wall_x_negative: Negative X boundary for walls (optional).
    - wall_y: Positive Y boundary for walls (optional).
    - wall_y_negative: Negative Y boundary for walls (optional).
    - wall_z: Height of walls in meters (optional).
    
    Returns a detailed message about the environment creation results.
    """
    try:
        blender = get_blender_connection()
        
        params = {
            "width": width,
            "depth": depth
        }
        if asset_id:
            params["asset_id"] = asset_id
        elif ground_description:
            params["ground_description"] = ground_description
        if categories_limitation:
            params["categories_limitation"] = categories_limitation
        
        # Wall parameters
        if wall_description:
            params["wall_description"] = wall_description
        if wall_asset_id:
            params["wall_asset_id"] = wall_asset_id
        if wall_categories_limitation:
            params["wall_categories_limitation"] = wall_categories_limitation
        if wall_x is not None:
            params["wall_x"] = wall_x
        if wall_x_negative is not None:
            params["wall_x_negative"] = wall_x_negative
        if wall_y is not None:
            params["wall_y"] = wall_y
        if wall_y_negative is not None:
            params["wall_y_negative"] = wall_y_negative
        if wall_z is not None:
            params["wall_z"] = wall_z
        
        result = blender.send_command("environment_artist", params)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if result.get("success"):
            message = result.get("message", "Environment design completed successfully")
            ground_asset_id = result.get("ground_asset_id", "")
            ground_texture_name = result.get("ground_texture_name", "")
            ground_combined_score = result.get("ground_combined_score")
            ground_llm_score = result.get("ground_llm_score")
            plane_name = result.get("plane_name", "ground_plane")
            x_times = result.get("x_times")
            y_times = result.get("y_times")
            side_length = result.get("side_length")
            width = result.get("width")
            depth = result.get("depth")
            
            output = f"{message}\n\n"
            output += f"=== Ground ===\n"
            output += f"Selected texture: {ground_asset_id}"
            if ground_texture_name:
                output += f" ({ground_texture_name})"
            output += "\n"
            if ground_combined_score is not None:
                output += f"Semantic match score: {ground_combined_score:.4f}\n"
            if ground_llm_score is not None:
                output += f"LLM relevance score: {ground_llm_score}/10\n"
            output += f"Ground plane created: {plane_name}\n"
            output += f"Dimensions: {width}m x {depth}m\n"
            if x_times and y_times:
                output += f"Tile configuration: {x_times}x{y_times} tiles\n"
            if side_length:
                output += f"Tile size: {side_length}m x {side_length}m\n"
            
            # Wall results
            wall_asset_id_result = result.get("wall_asset_id")
            if wall_asset_id_result:
                output += f"\n=== Walls ===\n"
                output += f"Wall texture: {wall_asset_id_result}"
                wall_texture_name = result.get("wall_texture_name", "")
                if wall_texture_name:
                    output += f" ({wall_texture_name})"
                output += "\n"
                walls_created = result.get("walls_created", [])
                if walls_created:
                    output += f"Walls created: {', '.join(walls_created)}\n"
                if result.get("roof_created"):
                    output += "Roof created: Yes\n"
            
            return output
        else:
            return f"Failed to create environment: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error in environment artist: {str(e)}")
        return f"Error in environment artist: {str(e)}"


@mcp.tool()
def format_assets(ctx: Context, path_to_script: str, model_output_dir: str) -> str:
    """
    Format all models in a storyboard script by resizing them based on provided dimensions,
    updating the asset_sheet with actual dimensions, and saving the formatted models.
    
    This tool reads a storyboard script JSON file, processes each model in the asset_sheet,
    calls format_asset to resize and anchor each model, updates the dimensions (width, depth, height),
    updates the main_file_path to the new saved location, and saves the updated JSON to the output directory.
    
    Parameters:
    - path_to_script: Absolute path to the storyboard script JSON file (e.g., /path/to/storyboard_snow_white.json)
    - model_output_dir: Absolute path to the directory where formatted models and updated JSON should be saved
    
    Returns a detailed message about the formatting results including success count and any errors.
    """
    try:
        blender = get_blender_connection()
        
        result = blender.send_command("format_assets", {
            "path_to_script": path_to_script,
            "model_output_dir": model_output_dir
        })
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if result.get("success"):
            formatted_count = result.get("formatted_count", 0)
            total_models = result.get("total_models", 0)
            formatted_models = result.get("formatted_models", [])
            output_script_path = result.get("output_script_path", "")
            errors = result.get("errors", [])
            error_count = result.get("error_count", 0)
            
            output = f"Model formatting completed!\n\n"
            output += f"Successfully formatted {formatted_count} out of {total_models} models:\n"
            for model_id in formatted_models:
                output += f"  - {model_id}\n"
            
            output += f"\nUpdated script saved to: {output_script_path}\n"
            
            if errors:
                output += f"\nEncountered {error_count} error(s):\n"
                for error in errors:
                    output += f"  - {error}\n"
            
            return output
        else:
            return f"Failed to format assets: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error in format_assets: {str(e)}")
        return f"Error in format_assets: {str(e)}"

@mcp.tool()
def import_asset_to_scene(
    ctx: Context,
    filepath: str,
    scene_name: str,
    location_x: Optional[float] = None,
    location_y: Optional[float] = None,
    location_z: Optional[float] = None,
    rotation_x: Optional[float] = None,
    rotation_y: Optional[float] = None,
    rotation_z: Optional[float] = None,
    scale_x: Optional[float] = None,
    scale_y: Optional[float] = None,
    scale_z: Optional[float] = None,
    dimensions_x: Optional[float] = None,
    dimensions_y: Optional[float] = None,
    dimensions_z: Optional[float] = None,
) -> str:
    """
    Import a GLB asset to a specific scene and apply transforms.
    
    Parameters:
    - filepath: File path to the GLB file.
    - scene_name: The name of the scene to import the model to (created if not exists).
    - location_x, location_y, location_z: Optional coordinates for location.
    - rotation_x, rotation_y, rotation_z: Optional rotation angles in XYZ Euler (radians).
    - scale_x, scale_y, scale_z: Optional scale factors.
    - dimensions_x, dimensions_y, dimensions_z: Optional dimensions.
    
    Returns a JSON string with success status and model information.
    """
    try:
        blender = get_blender_connection()
        
        # Build transform_parameters dict from individual parameters
        transform_parameters = {}
        
        # Location
        if any(v is not None for v in [location_x, location_y, location_z]):
            transform_parameters["location"] = {
                "x": location_x,
                "y": location_y,
                "z": location_z,
            }
        
        # Rotation
        if any(v is not None for v in [rotation_x, rotation_y, rotation_z]):
            transform_parameters["rotation"] = {
                "x": rotation_x,
                "y": rotation_y,
                "z": rotation_z,
            }
        
        # Scale
        if any(v is not None for v in [scale_x, scale_y, scale_z]):
            transform_parameters["scale"] = {
                "x": scale_x,
                "y": scale_y,
                "z": scale_z,
            }
        
        # Dimensions
        if any(v is not None for v in [dimensions_x, dimensions_y, dimensions_z]):
            transform_parameters["dimensions"] = {
                "x": dimensions_x,
                "y": dimensions_y,
                "z": dimensions_z,
            }
        
        result = blender.send_command("import_asset_to_scene", {
            "filepath": filepath,
            "scene_name": scene_name,
            "transform_parameters": transform_parameters if transform_parameters else None,
        })
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in import_asset_to_scene: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
def get_asset_transform(ctx: Context, model_name: str) -> str:
    """
    Get all transform properties of a model in Blender.
    
    Parameters:
    - model_name: Name of an existing model in the scene.
    
    Returns a JSON string with transform data including location, rotation, scale, and dimensions.
    """
    try:
        blender = get_blender_connection()
        
        result = blender.send_command("get_asset_transform", {
            "model_name": model_name,
        })
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_asset_transform: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
def transform_asset(
    ctx: Context,
    model_name: str,
    location_x: Optional[float] = None,
    location_y: Optional[float] = None,
    location_z: Optional[float] = None,
    rotation_x: Optional[float] = None,
    rotation_y: Optional[float] = None,
    rotation_z: Optional[float] = None,
    scale_x: Optional[float] = None,
    scale_y: Optional[float] = None,
    scale_z: Optional[float] = None,
    dimensions_x: Optional[float] = None,
    dimensions_y: Optional[float] = None,
    dimensions_z: Optional[float] = None,
) -> str:
    """
    Transform a model in Blender by setting its location, rotation, scale, and dimensions.
    
    Parameters:
    - model_name: Name of an existing model in the scene to transform.
    - location_x, location_y, location_z: Optional coordinates for location.
    - rotation_x, rotation_y, rotation_z: Optional rotation angles in XYZ Euler (radians).
    - scale_x, scale_y, scale_z: Optional scale factors.
    - dimensions_x, dimensions_y, dimensions_z: Optional dimensions.
    
    Returns a JSON string with success status and message.
    """
    try:
        blender = get_blender_connection()
        
        # Build parameter dicts from individual parameters
        location = None
        if any(v is not None for v in [location_x, location_y, location_z]):
            location = {"x": location_x, "y": location_y, "z": location_z}
        
        rotation = None
        if any(v is not None for v in [rotation_x, rotation_y, rotation_z]):
            rotation = {"x": rotation_x, "y": rotation_y, "z": rotation_z}
        
        scale = None
        if any(v is not None for v in [scale_x, scale_y, scale_z]):
            scale = {"x": scale_x, "y": scale_y, "z": scale_z}
        
        dimensions = None
        if any(v is not None for v in [dimensions_x, dimensions_y, dimensions_z]):
            dimensions = {"x": dimensions_x, "y": dimensions_y, "z": dimensions_z}
        
        result = blender.send_command("transform_asset", {
            "model_name": model_name,
            "location": location,
            "rotation": rotation,
            "scale": scale,
            "dimensions": dimensions,
        })
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in transform_asset: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
def import_all_assets_to_all_scenes_json_input(ctx: Context, json_filepath: str) -> str:
    """
    Import all assets to all scenes from a JSON file.
    
    The JSON file should contain:
    - asset_sheet: A list of asset dictionaries, each with 'id' and 'main_file_path'.
    - scene_details: A list of scene_detail dictionaries containing scene_id and scene_setup with layout info.
    
    Parameters:
    - json_filepath: Path to the JSON file containing asset_sheet and scene_details.
    
    Returns a JSON string with success status and any failed_objects list if failures occurred.
    """
    try:
        blender = get_blender_connection()
        
        result = blender.send_command("import_all_assets_to_all_scenes_json_input", {
            "json_filepath": json_filepath,
        })
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in import_all_assets_to_all_scenes_json_input: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
def switch_or_create_scene(ctx: Context, scene_name: str) -> str:
    """
    Switch to a scene with the given scene_name, or create it if it doesn't exist.
    
    Parameters:
    - scene_name: The name of the scene to switch to or create.
    
    Returns a JSON string with success status.
    """
    try:
        blender = get_blender_connection()
        
        result = blender.send_command("switch_or_create_scene", {
            "scene_name": scene_name,
        })
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in switch_or_create_scene: {str(e)}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
def delete_all_scenes_and_assets(ctx: Context) -> str:
    """
    Delete all scenes and assets, leaving only an empty scene named 'Scene'.
    
    This is useful for resetting Blender to a clean state before importing new assets.
    
    Returns a JSON string with success status, message, and list of deleted scenes.
    """
    try:
        blender = get_blender_connection()
        
        result = blender.send_command("delete_all_scenes_and_assets", {})
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in delete_all_scenes_and_assets: {str(e)}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
def import_supplementary_assets_to_all_scenes_json_input(ctx: Context, json_filepath: str) -> str:
    """
    Import supplementary assets to all scenes from a JSON file.
    
    The JSON file should contain:
    - asset_sheet: A list of supplementary asset dictionaries, each with 'asset_id' and 'main_file_path'.
    - scene_details: A list of scene_detail dictionaries containing scene_id and scene_setup with layout_description.
    
    Parameters:
    - json_filepath: Path to the JSON file containing asset_sheet and scene_details for supplementary assets.
    
    Returns a JSON string with success status and any failed_objects list if failures occurred.
    """
    try:
        blender = get_blender_connection()
        
        result = blender.send_command("import_supplementary_assets_to_all_scenes_json_input", {
            "json_filepath": json_filepath,
        })
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in import_supplementary_assets_to_all_scenes_json_input: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
def import_animated_assets_to_all_shots_json_input(ctx: Context, json_filepath: str) -> str:
    """
    Import animated assets to all shots from a JSON file.
    
    This function reads a story script JSON file and:
    1. Loops through shot_details
    2. For each shot, creates a linked copy of Scene_{scene_id} named Scene_{scene_id}_Shot_{shot_id}
    3. Imports animated models with matching transforms from the original scene
    
    Parameters:
    - json_filepath: Path to the JSON file containing shot_details and scene_details.
    
    Returns a JSON string with success status and details about the import.
    """
    try:
        blender = get_blender_connection()
        
        result = blender.send_command("import_animated_assets_to_all_shots_json_input", {
            "json_filepath": json_filepath,
        })
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in import_animated_assets_to_all_shots_json_input: {str(e)}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
def delete_all_shots(ctx: Context) -> str:
    """
    Delete all shot scenes in Blender.
    
    Deletes all scenes matching pattern Scene_{scene_id}_Shot_{shot_id},
    but leaves all other scenes intact (including original scenes like Scene_{scene_id}).
    
    Returns a JSON string with success status and list of deleted scenes.
    """
    try:
        blender = get_blender_connection()
        
        result = blender.send_command("delete_all_shots", {})
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in delete_all_shots: {str(e)}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
def delete_asset(ctx: Context, model_name: str) -> str:
    """
    Delete an asset (object) from the current scene by its name.
    
    Parameters:
    - model_name: The name of the object to delete.
    
    Returns a JSON string with success status and message.
    """
    try:
        blender = get_blender_connection()
        
        result = blender.send_command("delete_asset", {
            "model_name": model_name
        })
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in delete_asset: {str(e)}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
def set_render(ctx: Context, engine: str = "EEVEE", samples: Optional[int] = None, persistent_data: bool = True) -> str:
    """
    Configure render settings for all scenes in the Blender file.
    
    Parameters:
    - engine: Render engine to use. Options: "EEVEE" or "Cycles" (case-insensitive).
    - samples: Number of render samples (optional).
              For EEVEE: default is 64
              For Cycles: default is 512 (render), viewport uses 128
    - persistent_data: Whether to enable persistent data for faster re-renders. Default: True.
    
    Returns a JSON string with success status and details about configured scenes.
    """
    try:
        blender = get_blender_connection()
        
        params = {"engine": engine, "persistent_data": persistent_data}
        if samples is not None:
            params["samples"] = samples
        
        result = blender.send_command("set_render", params)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if result.get("success"):
            message = result.get("message", "Render settings configured successfully")
            scenes = result.get("scenes", [])
            
            output = f"{message}\n\n"
            for scene_info in scenes:
                output += f"Scene: {scene_info.get('scene')}\n"
                output += f"  Engine: {scene_info.get('engine')}\n"
                if scene_info.get('device'):
                    output += f"  Device: {scene_info.get('device')}\n"
                output += f"  Render samples: {scene_info.get('render_samples')}\n"
                if scene_info.get('viewport_samples'):
                    output += f"  Viewport samples: {scene_info.get('viewport_samples')}\n"
                if scene_info.get('render_noise_threshold'):
                    output += f"  Render noise threshold: {scene_info.get('render_noise_threshold')}\n"
                if scene_info.get('viewport_noise_threshold'):
                    output += f"  Viewport noise threshold: {scene_info.get('viewport_noise_threshold')}\n"
                output += f"  Persistent data: {scene_info.get('persistent_data')}\n"
            
            return output
        else:
            return f"Failed to set render: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error in set_render: {str(e)}")
        return f"Error in set_render: {str(e)}"


@mcp.tool()
def get_camera_info(
    ctx: Context,
    scene_name: str,
    camera_name: str,
) -> str:
    """
    Get camera information from Blender by scene name and camera name.
    
    Reads camera transform, focal length, DoF settings, and keyframe data.
    
    Parameters:
    - scene_name: Name of the scene containing the camera
    - camera_name: Name of the camera object
    
    Returns a JSON string with camera info including transforms, parameters, and animation data.
    """
    try:
        blender = get_blender_connection()
        
        result = blender.send_command("get_camera_info", {
            "scene_name": scene_name,
            "camera_name": camera_name,
        })
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in get_camera_info: {str(e)}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
def camera_operator_tool(
    ctx: Context,
    path_to_input_json: str,
    vision_model: str = "gemini/gemini-2.5-flash",
    anyllm_api_key: Optional[str] = None,
    anyllm_api_base: Optional[str] = None,
    camera_type: str = "director",
    max_additional_cameras: int = 1,
    camera_name_filter: Optional[List[str]] = None,
    start_frame: int = 1,
    end_frame: int = 73,
    max_adjustment_rounds: int = 5,
    preview_image_save_dir: Optional[str] = None,
) -> str:
    """
    Place cameras in Blender scenes based on storyboard instructions.
    
    Parameters:
    - path_to_input_json: Path to JSON file with shot_details
    - vision_model: Vision model for LLM (default: gemini/gemini-2.5-flash)
    - anyllm_api_key: API key for any-llm
    - anyllm_api_base: API base URL for any-llm (optional)
    - camera_type: 'director', 'additional', or 'all' (default: director)
    - max_additional_cameras: Maximum additional cameras per shot (default: 1)
    - camera_name_filter: List of camera names to place (None = all)
    - start_frame: Start frame for camera animation (default: 1)
    - end_frame: End frame for camera animation (default: 73)
    - max_adjustment_rounds: Maximum LLM adjustment rounds (default: 5)
    - preview_image_save_dir: Directory to save preview images (optional)
    
    Returns a JSON string with shot_details, cameras_placed, cameras_failed.
    """
    try:
        blender = get_blender_connection()
        
        result = blender.send_command("camera_operator", {
            "path_to_input_json": path_to_input_json,
            "vision_model": vision_model,
            "anyllm_api_key": anyllm_api_key,
            "anyllm_api_base": anyllm_api_base,
            "camera_type": camera_type,
            "max_additional_cameras": max_additional_cameras,
            "camera_name_filter": camera_name_filter,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "max_adjustment_rounds": max_adjustment_rounds,
            "preview_image_save_dir": preview_image_save_dir,
        })
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in camera_operator: {str(e)}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
def resume_camera_operator_tool(
    ctx: Context,
    path_to_input_json: str,
    camera_name_filter: Optional[List[str]] = None,
) -> str:
    """
    Resume/recreate cameras from a previously saved JSON file without LLM.
    
    Parameters:
    - path_to_input_json: Path to JSON file containing camera placement info
    - camera_name_filter: List of camera names to resume (None = all)
    
    Returns a JSON string with cameras_resumed, cameras_failed.
    """
    try:
        blender = get_blender_connection()
        
        result = blender.send_command("resume_camera_operator", {
            "path_to_input_json": path_to_input_json,
            "camera_name_filter": camera_name_filter,
        })
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in resume_camera_operator: {str(e)}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)


# Main execution

def main():
    """Run the MCP server"""
    mcp.run()

if __name__ == "__main__":
    main()