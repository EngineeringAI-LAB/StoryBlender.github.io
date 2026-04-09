import os
import json
import glob
import re
import gradio as gr
from .blender_client import BlenderClient
from .path_utils import make_paths_absolute


def apply_asset_modifications(blender_client, project_dir):
    """Apply asset modifications from the latest layout_script JSON.
    
    Args:
        blender_client: BlenderClient instance for communicating with Blender
        project_dir: Project directory path
        
    Returns:
        dict: Result with success/error and data
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path"
        }
    
    # Find the latest layout_script JSON
    layout_script_dir = os.path.join(project_dir, "layout_script")
    if not os.path.isdir(layout_script_dir):
        return {
            "error": f"⚠️ Layout script directory not found: {layout_script_dir}"
        }
    
    # Find all layout_script_v*.json files
    pattern = os.path.join(layout_script_dir, "layout_script_v*.json")
    files = glob.glob(pattern)
    
    if not files:
        return {
            "error": "⚠️ No layout_script JSON found. Please generate a layout script first (Step 3)."
        }
    
    # Sort by version number and get the latest
    def extract_version(filepath):
        match = re.search(r'layout_script_v(\d+)\.json$', filepath)
        return int(match.group(1)) if match else 0
    
    files.sort(key=extract_version)
    json_filepath = files[-1]  # Latest version
    
    # Ensure MCP server is running
    success, message = blender_client.ensure_server_running()
    if not success:
        return {
            "error": f"⚠️ {message}"
        }
    
    # Load JSON and convert relative paths to absolute for Blender
    temp_input_path = None
    try:
        with open(json_filepath, 'r') as f:
            input_data = json.load(f)
        input_data = make_paths_absolute(input_data, project_dir)
        
        temp_input_path = json_filepath + ".tmp"
        with open(temp_input_path, 'w') as f:
            json.dump(input_data, f, indent=2)
    except Exception as e:
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return {
            "error": f"⚠️ Failed to prepare input for Blender: {str(e)}"
        }
    
    # Call apply_asset_modifications_json_input via BlenderClient
    try:
        response = blender_client.apply_asset_modifications_json_input(
            json_filepath=temp_input_path
        )
        
        if response.get("status") == "error":
            return {
                "error": f"⚠️ Blender error: {response.get('message', 'Unknown error')}"
            }
        
        result = response.get("result", {})
        
        if result.get("success"):
            modified_count = result.get("modified_count", 0)
            scenes_created = result.get("scenes_created", 0)
            msg_parts = [f"✅ Asset modifications applied successfully!"]
            if scenes_created > 0:
                msg_parts.append(f"Created {scenes_created} shot scene(s).")
            msg_parts.append(f"Modified {modified_count} asset(s).")
            return {
                "success": True,
                "message": " ".join(msg_parts),
                "json_filepath": json_filepath,
            }
        else:
            errors = result.get("errors", [])
            error_msg = result.get("error", "")
            
            if errors:
                error_lines = ["⚠️ Some asset modifications failed:"]
                for err in errors[:10]:  # Limit to first 10 errors
                    error_lines.append(f"  - {err}")
                if len(errors) > 10:
                    error_lines.append(f"  ... and {len(errors) - 10} more errors")
                return {
                    "error": "\n".join(error_lines),
                    "errors": errors,
                }
            elif error_msg:
                return {
                    "error": f"⚠️ {error_msg}"
                }
            else:
                return {
                    "error": "⚠️ Apply modifications failed with unknown error"
                }
                
    except Exception as e:
        return {
            "error": f"⚠️ Failed to apply asset modifications: {str(e)}"
        }
    finally:
        # Clean up temporary input file
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError:
                pass


def show_loading_and_apply_modifications(blender_client, project_dir):
    """Show loading state while applying asset modifications."""
    # Loading state
    loading_state = (
        gr.update(value="🔄 **Applying asset modifications...** Please wait.", visible=True),
        gr.update(visible=False),  # Hide apply button
    )
    
    yield loading_state
    
    # Apply modifications
    result = apply_asset_modifications(blender_client, project_dir)
    
    # Final state
    if result.get("success"):
        success_msg = result.get("message", "✅ Asset modifications applied successfully!")
        yield (
            gr.update(value=success_msg, visible=True),
            gr.update(visible=True),  # Show apply button
        )
    else:
        error_msg = result.get("error", "⚠️ Unknown error occurred")
        yield (
            gr.update(value=error_msg, visible=True),
            gr.update(visible=True),  # Show apply button
        )


def create_apply_modifications_wrapper(blender_client):
    """Factory function to create an apply modifications wrapper bound to specific components."""
    def apply_wrapper(project_dir):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_apply_modifications(blender_client, project_dir):
            yield result
    return apply_wrapper


def create_apply_asset_modifications_ui(project_dir, blender_client):
    """Create the Step 8: Apply Asset Modifications in All Shots UI section.
    
    Args:
        project_dir: Gradio component for project directory
        blender_client: BlenderClient instance for communicating with Blender
    
    Returns:
        dict with UI components that may be needed by other parts of the app
    """
    gr.Markdown("## Step 8: Apply Asset Modifications in All Shots")
    gr.Markdown("Apply add/remove/transform modifications to objects in shot scenes based on the latest layout_script.")
    
    # Input info
    gr.Markdown("**Input:** `project_dir/layout_script/layout_script_v{N}.json` (latest)")
    
    # Status indicator
    apply_modifications_status = gr.Markdown(value="", visible=False)
    
    # Apply Asset Modifications button
    apply_modifications_btn = gr.Button(
        "🔧 Apply Asset Modifications",
        variant="primary",
        size="lg"
    )
    
    # Create wrapper function for the button
    apply_modifications_wrapper = create_apply_modifications_wrapper(blender_client)
    
    # Apply Asset Modifications button click handler
    apply_modifications_btn.click(
        fn=apply_modifications_wrapper,
        inputs=[project_dir],
        outputs=[
            apply_modifications_status,
            apply_modifications_btn,
        ],
    )
    
    return {
        "apply_modifications_status": apply_modifications_status,
        "apply_modifications_btn": apply_modifications_btn,
    }
