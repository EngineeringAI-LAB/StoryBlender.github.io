from pydantic import RootModel, BaseModel, Field
from typing import Optional, Dict, List, Any
import json
import base64
import mimetypes
import os
import gc
import warnings

from collections import Counter
from copy import deepcopy

try:
    from ..llm_completion import completion
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from llm_completion import completion


class AssetDimension(BaseModel):
    """Schema for a single asset dimension estimation.
    
    Only ONE of width, depth, or height should be provided (the most confident estimate).
    The other two must be null.
    """
    asset_id: str = Field(description="Asset ID from the asset sheet")
    width: Optional[float] = Field(None, description="Width dimension in meters (X-axis in Blender), accurate to 2 decimal places")
    depth: Optional[float] = Field(None, description="Depth dimension in meters (Y-axis in Blender), accurate to 2 decimal places")
    height: Optional[float] = Field(None, description="Height dimension in meters (Z-axis in Blender), accurate to 2 decimal places")


class AssetDimensionEstimation(RootModel):
    """Schema for the complete dimension estimation output.
    
    A JSON array containing dimension estimates for all assets.
    """
    root: List[AssetDimension] = Field(description="List of asset dimension estimations")

def process_url_or_path(url_or_path: str) -> str:
    """Process a URL or local file path and return a URL suitable for API calls.
    
    Args:
        url_or_path: Either a web URL or a path to a local image file.
        
    Returns:
        The URL as-is if it's a web URL, or a base64 data URL if it's a local path.
    """
    if url_or_path.startswith(('http://', 'https://')):
        return url_or_path
    
    # It's a local file path, convert to base64 data URL
    if not os.path.isfile(url_or_path):
        raise FileNotFoundError(f"File not found: {url_or_path}")
    
    mime_type, _ = mimetypes.guess_type(url_or_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    
    with open(url_or_path, 'rb') as f:
        image_data = f.read()
    
    base64_data = base64.b64encode(image_data).decode('utf-8')
    return f"data:{mime_type};base64,{base64_data}"

def create_dimension_estimation_prompt(concept_data: dict) -> List[Dict[str, Any]]:
    """Convert concept artist output to a multimodal prompt for dimension estimation.
    
    Args:
        concept_data: Dictionary output from concept artist agent
        
    Returns:
        A list of dicts in any-llm format with "type" and "text"/"image_url" keys
        suitable for ``generate_asset_dimension_estimation``.
    """
    prompt_parts: List[Dict[str, Any]] = []
    
    # 1. Process storyboard_outline (include everything)
    if "storyboard_outline" in concept_data:
        md_text = "# Storyboard Outline\n\n"
        for scene in concept_data["storyboard_outline"]:
            md_text += f"## Scene {scene.get('scene_id', 'N/A')}\n\n"
            md_text += f"{scene.get('scene_description', '')}\n\n"
            
            if "shots" in scene:
                md_text += "### Shots\n\n"
                for shot in scene["shots"]:
                    md_text += f"- **Shot {shot.get('shot_id', 'N/A')}**: {shot.get('shot_description', '')}\n"
                md_text += "\n"
        
        prompt_parts.append({"type": "text", "text": md_text})
    
    # 2. Process scene_details (only scene_id, scene_setup, asset_ids, layout_description)
    if "scene_details" in concept_data:
        md_text = "# Scene Details\n\n"
        for scene in concept_data["scene_details"]:
            md_text += f"## Scene {scene.get('scene_id', 'N/A')}\n\n"
            
            scene_setup = scene.get("scene_setup", {})
            
            # asset_ids
            if "asset_ids" in scene_setup:
                md_text += f"**Asset IDs**: {', '.join(scene_setup['asset_ids'])}\n\n"
            
            # layout_description
            if "layout_description" in scene_setup:
                md_text += f"**Layout Description**: {scene_setup['layout_description']}\n\n"
        
        prompt_parts.append({"type": "text", "text": md_text})
    
    # 3. Process asset_sheet (asset_id, description, and three orthographic views) - put at the end
    if "asset_sheet" in concept_data:
        md_text = "# Asset Sheet\n\n"
        
        for asset in concept_data["asset_sheet"]:
            asset_id = asset.get("asset_id", "N/A")
            description = asset.get("description", "")
            front_view_url = asset.get("front_view_url", "")
            top_view_url = asset.get("top_view_url", "")
            left_view_url = asset.get("left_view_url", "")
            
            md_text += f"## {asset_id}\n\n"
            md_text += f"{description}\n\n"
            
            # Add images with labels for each orthographic view
            # Front view (shows width and height)
            if front_view_url:
                md_text += "**Front View** (shows width on horizontal axis, height on vertical axis):\n\n"
                prompt_parts.append({"type": "text", "text": md_text})
                md_text = ""
                try:
                    prompt_parts.append({
                        "type": "image_url",
                        "image_url": {"url": process_url_or_path(front_view_url)}
                    })
                except Exception as exc:
                    print(f"Warning: Failed to load front view image for {asset_id}: {exc}")
            
            # Top view (shows width and depth)
            if top_view_url:
                md_text += "\n**Top View** (shows width on horizontal axis, depth on vertical axis):\n\n"
                prompt_parts.append({"type": "text", "text": md_text})
                md_text = ""
                try:
                    prompt_parts.append({
                        "type": "image_url",
                        "image_url": {"url": process_url_or_path(top_view_url)}
                    })
                except Exception as exc:
                    print(f"Warning: Failed to load top view image for {asset_id}: {exc}")
            
            # Left view (shows depth and height)
            if left_view_url:
                md_text += "\n**Left View** (shows depth on horizontal axis, height on vertical axis):\n\n"
                prompt_parts.append({"type": "text", "text": md_text})
                md_text = ""
                try:
                    prompt_parts.append({
                        "type": "image_url",
                        "image_url": {"url": process_url_or_path(left_view_url)}
                    })
                except Exception as exc:
                    print(f"Warning: Failed to load left view image for {asset_id}: {exc}")
            
            # Add any remaining text
            if md_text:
                prompt_parts.append({"type": "text", "text": md_text})
                md_text = ""
    
    return prompt_parts


def create_supplementary_dimension_estimation_prompt(
    layout_script: dict,
    supplementary_assets: dict
) -> List[Dict[str, Any]]:
    """Convert layout script and supplementary assets to a multimodal prompt for dimension estimation.
    
    Args:
        layout_script: Dictionary containing the full story script with concept_data,
            storyboard_outline, scene_details with layout info, and asset_sheet with dimensions.
        supplementary_assets: Dictionary containing new supplementary assets to estimate,
            with asset_sheet and scene_details for where they should be placed.
        
    Returns:
        A list of dicts in any-llm format with "type" and "text"/"image_url" keys
        suitable for ``generate_asset_dimension_estimation`` with estimation_type="supplementary_assets".
    """
    prompt_parts: List[Dict[str, Any]] = []
    
    # 1. Process storyboard_outline from layout_script (include everything)
    if "storyboard_outline" in layout_script:
        md_text = "# Storyboard Outline\n\n"
        for scene in layout_script["storyboard_outline"]:
            md_text += f"## Scene {scene.get('scene_id', 'N/A')}\n\n"
            md_text += f"{scene.get('scene_description', '')}\n\n"
            
            if "shots" in scene:
                md_text += "### Shots\n\n"
                for shot in scene["shots"]:
                    md_text += f"- **Shot {shot.get('shot_id', 'N/A')}**: {shot.get('shot_description', '')}\n"
                md_text += "\n"
        
        prompt_parts.append({"type": "text", "text": md_text})
    
    # 2. Process scene_details from layout_script as "# Current Scene Details"
    if "scene_details" in layout_script:
        md_text = "# Current Scene Details\n\n"
        for scene in layout_script["scene_details"]:
            md_text += f"## Scene {scene.get('scene_id', 'N/A')}\n\n"
            
            scene_setup = scene.get("scene_setup", {})
            
            # asset_ids
            if "asset_ids" in scene_setup:
                md_text += f"**Asset IDs**: {', '.join(scene_setup['asset_ids'])}\n\n"
            
            # layout_description
            if "layout_description" in scene_setup:
                layout_desc = scene_setup["layout_description"]
                if isinstance(layout_desc, dict):
                    md_text += f"**Layout Description**: {json.dumps(layout_desc, indent=2)}\n\n"
                else:
                    md_text += f"**Layout Description**: {layout_desc}\n\n"
        
        prompt_parts.append({"type": "text", "text": md_text})
    
    # 3. Process asset_sheet from layout_script as "# Existing Assets Sheet" (only dimensions, no description/thumbnail)
    if "asset_sheet" in layout_script:
        md_text = "# Existing Assets Sheet\n\n"
        md_text += "| asset_id | width | depth | height |\n"
        md_text += "|----------|-------|-------|--------|\n"
        
        for asset in layout_script["asset_sheet"]:
            asset_id = asset.get("asset_id", "N/A")
            width = asset.get("width", "N/A")
            depth = asset.get("depth", "N/A")
            height = asset.get("height", "N/A")
            md_text += f"| {asset_id} | {width} | {depth} | {height} |\n"
        
        md_text += "\n"
        prompt_parts.append({"type": "text", "text": md_text})
    
    # 4. Process asset_sheet from supplementary_assets as "# New Supplementary Assets to be Estimated"
    if "asset_sheet" in supplementary_assets:
        md_text = "# New Supplementary Assets to be Estimated\n\n"
        
        for asset in supplementary_assets["asset_sheet"]:
            asset_id = asset.get("asset_id", "N/A")
            description = asset.get("description", "")
            front_view_url = asset.get("front_view_url", "")
            top_view_url = asset.get("top_view_url", "")
            left_view_url = asset.get("left_view_url", "")
            
            md_text += f"## {asset_id}\n\n"
            md_text += f"{description}\n\n"
            
            # Add images with labels for each orthographic view
            # Front view (shows width and height)
            if front_view_url:
                md_text += "**Front View** (shows width on horizontal axis, height on vertical axis):\n\n"
                prompt_parts.append({"type": "text", "text": md_text})
                md_text = ""
                try:
                    prompt_parts.append({
                        "type": "image_url",
                        "image_url": {"url": process_url_or_path(front_view_url)}
                    })
                except Exception as exc:
                    print(f"Warning: Failed to load front view image for {asset_id}: {exc}")
            
            # Top view (shows width and depth)
            if top_view_url:
                md_text += "\n**Top View** (shows width on horizontal axis, depth on vertical axis):\n\n"
                prompt_parts.append({"type": "text", "text": md_text})
                md_text = ""
                try:
                    prompt_parts.append({
                        "type": "image_url",
                        "image_url": {"url": process_url_or_path(top_view_url)}
                    })
                except Exception as exc:
                    print(f"Warning: Failed to load top view image for {asset_id}: {exc}")
            
            # Left view (shows depth and height)
            if left_view_url:
                md_text += "\n**Left View** (shows depth on horizontal axis, height on vertical axis):\n\n"
                prompt_parts.append({"type": "text", "text": md_text})
                md_text = ""
                try:
                    prompt_parts.append({
                        "type": "image_url",
                        "image_url": {"url": process_url_or_path(left_view_url)}
                    })
                except Exception as exc:
                    print(f"Warning: Failed to load left view image for {asset_id}: {exc}")
            
            # Add any remaining text
            if md_text:
                prompt_parts.append({"type": "text", "text": md_text})
                md_text = ""
    
    # 5. Process scene_details from supplementary_assets as "# New Scene Details with Supplementary Assets"
    if "scene_details" in supplementary_assets:
        md_text = "# New Scene Details with Supplementary Assets\n\n"
        for scene in supplementary_assets["scene_details"]:
            md_text += f"## Scene {scene.get('scene_id', 'N/A')}\n\n"
            
            scene_setup = scene.get("scene_setup", {})
            md_text += f"**Scene Setup**:\n```json\n{json.dumps(scene_setup, indent=2)}\n```\n\n"
        
        prompt_parts.append({"type": "text", "text": md_text})
    
    return prompt_parts


def validate_output(concept_data: dict, asset_dimension_estimation: List[Dict]) -> bool:
    """Validate that asset IDs in concept_data and estimation output match exactly.

    Args:
        concept_data: The full concept artist output containing an `asset_sheet` list.
        asset_dimension_estimation: The list output from `generate_asset_dimension_estimation`.

    Returns:
        True if the IDs match exactly in names and counts, False otherwise.
    """
    try:
        if not concept_data or not isinstance(asset_dimension_estimation, list):
            return False

        # Extract expected IDs from asset_sheet
        asset_sheet = concept_data.get("asset_sheet", [])
        expected_ids = [a.get("asset_id") for a in asset_sheet if isinstance(a, dict) and a.get("asset_id")]

        # Extract produced IDs from estimation output
        produced_ids = [e.get("asset_id") for e in asset_dimension_estimation if isinstance(e, dict) and e.get("asset_id")]

        # Compare multisets to ensure exact name and number match (accounts for duplicates)
        return Counter(expected_ids) == Counter(produced_ids)
    except Exception:
        return False

def merge_estimation(concept_data: dict, estimation: List[Dict]) -> dict:
    """Merge width/depth/height from estimation into concept_data.asset_sheet by asset_id.

    Returns a merged copy of concept_data without mutating the input.
    """
    if not isinstance(concept_data, dict) or not isinstance(estimation, list):
        return concept_data

    merged = deepcopy(concept_data)
    asset_sheet = merged.get("asset_sheet", [])

    # Build asset_id -> dims map
    id_to_dims = {}
    for item in estimation:
        if not isinstance(item, dict):
            continue
        _id = item.get("asset_id")
        if not _id:
            continue
        id_to_dims[_id] = {
            "width": item.get("width"),
            "depth": item.get("depth"),
            "height": item.get("height"),
        }

    # Merge into asset_sheet
    if isinstance(asset_sheet, list):
        for asset in asset_sheet:
            if not isinstance(asset, dict):
                continue
            aid = asset.get("asset_id")
            if aid and aid in id_to_dims:
                dims = id_to_dims[aid]
                asset["width"] = dims.get("width")
                asset["depth"] = dims.get("depth")
                asset["height"] = dims.get("height")

    return merged

def generate_asset_dimension_estimation(
    anyllm_api_key=None,
    anyllm_api_base=None,
    anyllm_provider="gemini",
    reasoning_model="gemini-3.1-pro-preview",
    contents=None,
    concept_data=None,
    response_schema=AssetDimensionEstimation,
    reasoning_effort="high",
    estimation_type="core_assets"
):
    """Generate a structured JSON array of asset dimension estimates with any-llm.

    This calls the model with a system instruction and the provided multimodal
    ``contents`` (e.g., text and image dicts from ``create_dimension_estimation_prompt``),
    and parses the JSON response into a Python object.

    Args:
        anyllm_api_key: Optional API key used to authenticate the client.
        anyllm_api_base: Optional base URL for the API service.
        anyllm_provider: LLM provider (default: "gemini").
        reasoning_model: The model name to invoke, e.g. "gemini-3.1-pro-preview".
        contents: The input message contents to send to the model. A list of
            dicts with "type" and "text"/"image_url" keys as produced by
            ``create_dimension_estimation_prompt``.
        concept_data: The full concept artist output containing an `asset_sheet` list.
        response_schema: A Pydantic model (RootModel) describing the expected JSON shape.
            Defaults to ``AssetDimensionEstimation`` to enforce a list of asset-dimension
            objects.
        reasoning_effort: Controls thinking capability. Options are "low",
            "medium", or "high". Defaults to "high".
        estimation_type: Type of estimation to perform. Options are "core_assets" 
            (default) or "supplementary_assets". Use "supplementary_assets" when 
            estimating dimensions for supplementary assets that populate the scene.

    Returns:
        list[dict]: The parsed JSON array of objects compliant with ``response_schema``,
            typically a list like [{"asset_id": str, "width": float|None, "depth": float|None,
            "height": float|None}].
    """
    
    system_instruction = """
You are an expert 3D object dimension estimator for Blender. Your sole task is to analyze a storyboard script and its accompanying asset information (including orthographic view images) to determine the most likely real-world scale for each 3D asset.

INPUT:
You will be provided with a complete storyboard script in Markdown format. This script contains three main sections:
Storyboard Outline: This details the story's plot, scene-by-scene descriptions, and individual shot descriptions. Use this to understand the narrative context.
Asset Sheet: This is a list of all 3D assets used in the story. Each asset has an asset_id, a description, and THREE orthographic view images:
  - **Front View**: Shows the asset from the front. The horizontal axis represents WIDTH (X-axis in Blender), and the vertical axis represents HEIGHT (Z-axis in Blender).
  - **Top View**: Shows the asset from above. The horizontal axis represents WIDTH (X-axis in Blender), and the vertical axis represents DEPTH (Y-axis in Blender).
  - **Left View**: Shows the asset from the left side. The horizontal axis represents DEPTH (Y-axis in Blender), and the vertical axis represents HEIGHT (Z-axis in Blender).
You MUST analyze these orthographic views to accurately understand the asset's shape, proportions, and which dimension is most prominent.
Scene Details: This section provides a setup for each scene, including which asset_ids appear and a crucial layout_description. You MUST pay close attention to the layout_description as it describes the spatial relationships between objects, providing critical clues for relative scale.

REASONING PROCESS:
To determine the dimensions, you must synthesize information from all provided sources:
Common Knowledge: Use real-world knowledge. (e.g., An adult human's height is typically between 1.5m and 1.9m. A 'red_apple' width is around 0.08m. A 'goblet' is a handheld object.)
Visual Analysis (Orthographic Views): Analyze the three orthographic views to understand the asset's true 3D proportions:
  - Use the **Front View** to compare width vs height - identify if the object is tall/short or wide/narrow from the front.
  - Use the **Top View** to compare width vs depth - identify if the object is elongated in width or depth when viewed from above.
  - Use the **Left View** to compare depth vs height - cross-verify the height and understand the depth profile.
  - By combining these views, you can accurately determine which dimension (width, depth, or height) is the most prominent and easiest to estimate with confidence.
For 3D models that are approximately cuboid in shape and where the definition of the 'front' may be ambiguous (e.g., tables, houses, carpets, where the front and back may appear identical), the dimensions are defined as follows: width denotes the long side, depth the short side, and height the vertical height. Providing just one of these is sufficient.
Contextual & Relational Clues (Layout Descriptions): Use the scene layouts to infer relative size.
If 'Snow White' (a person) stands 'in front of' the 'dwarfs_cottage' (Scene 3), the cottage's height or width must be relative to her height.
If the 'Old Hag' is 'holding the red apple' (Scene 5), the apple must be hand-sized.
If the 'Evil Queen' stands 'facing the Magic Mirror' (Scene 1), the mirror's height and width should be appropriate for a person to stand in front of, likely taller than a person.
If a 'jeweled_box' is 'on the ground near the Huntsman' (Scene 2), it must be relatively small, consistent with its description.

OUTPUT REQUIREMENTS:
Your entire response MUST be a JSON array, and nothing else. Do not provide any text, explanation, or conversational wrapper before or after the JSON array.
The JSON array must contain one object for every asset listed in the asset_sheet.
Each asset object MUST follow this exact format:
{"asset_id": "asset_id", "width": number | null, "depth": number | null, "height": number | null}
For each asset, you MUST determine its primary spatial dimension in Blender (width=X, depth=Y, height=Z).
You MUST provide ONLY ONE of these three values: width, depth, or height. This must be the single dimension you have the highest confidence in estimating.
The other two dimension fields for that asset MUST be set to null.
The provided dimension MUST be a number in meters.
The number MUST be accurate to two decimal places (e.g., 1.70, 0.45).

EXAMPLE:
For "snow_white" (a person), height is the most confident dimension.
For "jeweled_box" (a small prop), width or depth might be most confident based on the thumbnail.
For "dwarfs_cottage" (a building), height or width would be a good choice.
A correct output based on this logic would look like this:
[{"asset_id":"snow_white", "width": null, "depth": null, "height": 1.54}, {"asset_id":"jeweled_box", "width": 0.30, "depth": null, "height": null}, {"asset_id":"dwarfs_cottage", "width": null, "depth": null, "height": 3.50}]
Begin your analysis after the user input. Your output must be only the JSON array."""

    system_instruction_for_supplementary_assets = """
You are an expert 3D object dimension estimator for Blender. Your sole task is to analyze a storyboard script and estimate the dimensions of NEW SUPPLEMENTARY ASSETS that will populate the scene and make it more immersive.

INPUT:
You will be provided with a storyboard script in Markdown format containing these sections:
1. Storyboard Outline: The story's plot, scene descriptions, and shot descriptions for narrative context.
2. Current Scene Details: The existing scene setup with core asset_ids and layout_description showing spatial relationships.
3. Existing Assets Sheet: A table of ALREADY ESTIMATED core assets with their asset_id, width, depth, and height. These dimensions are FIXED references - DO NOT re-estimate them.
4. New Supplementary Assets to be Estimated: The NEW assets you must estimate. Each has an asset_id, description, and THREE orthographic view images:
   - **Front View**: Shows the asset from the front. The horizontal axis represents WIDTH (X-axis in Blender), and the vertical axis represents HEIGHT (Z-axis in Blender).
   - **Top View**: Shows the asset from above. The horizontal axis represents WIDTH (X-axis in Blender), and the vertical axis represents DEPTH (Y-axis in Blender).
   - **Left View**: Shows the asset from the left side. The horizontal axis represents DEPTH (Y-axis in Blender), and the vertical axis represents HEIGHT (Z-axis in Blender).
5. New Scene Details with Supplementary Assets: Where each new supplementary asset should be placed relative to existing assets.

CRITICAL RULES:
- You must ONLY estimate dimensions for assets listed in "New Supplementary Assets to be Estimated".
- DO NOT include any assets from "Existing Assets Sheet" in your output.
- Use the existing asset dimensions as REFERENCE for scale. For example, if a 'candelabra' is placed next to a 'magic_mirror' (height: 1.80m), the candelabra should be appropriately sized relative to that mirror.
- For 3D models that are approximately cuboid in shape and where the definition of the 'front' may be ambiguous (e.g., tables, houses, carpets, where the front and back may appear identical), the dimensions are defined as follows: width denotes the long side, depth the short side, and height the vertical height. Providing just one of these is sufficient.

REASONING PROCESS:
To determine the dimensions of the NEW supplementary assets:
1. Common Knowledge: Use real-world knowledge. (e.g., A floor candelabra is typically 1.2-1.5m tall. A wooden bucket is about 0.3-0.4m tall.)
2. Visual Analysis (Orthographic Views): Analyze the three orthographic views to understand the asset's true 3D proportions:
   - Use the **Front View** to compare width vs height - identify if the object is tall/short or wide/narrow from the front.
   - Use the **Top View** to compare width vs depth - identify if the object is elongated in width or depth when viewed from above.
   - Use the **Left View** to compare depth vs height - cross-verify the height and understand the depth profile.
   - By combining these views, you can accurately determine which dimension (width, depth, or height) is the most prominent and easiest to estimate with confidence.
3. Contextual & Relational Clues: Use the "New Scene Details with Supplementary Assets" to understand placement. If a 'crimson_runner_rug' is placed 'in_front_of' the 'magic_mirror' at distance 1.5m, it should be sized appropriately to fit that space and look natural relative to the mirror's width.
4. Reference Existing Dimensions: Cross-reference with the "Existing Assets Sheet" to ensure new assets are scaled correctly relative to established objects.

OUTPUT REQUIREMENTS:
Your entire response MUST be a JSON array, and nothing else.
The JSON array must contain one object for EVERY asset listed in "New Supplementary Assets to be Estimated" ONLY.
DO NOT include any assets from "Existing Assets Sheet".
Each asset object MUST follow this exact format:
{"asset_id": "asset_id", "width": number | null, "depth": number | null, "height": number | null}
For each asset, you MUST determine its primary spatial dimension in Blender (width=X, depth=Y, height=Z).
You MUST provide ONLY ONE of these three values: width, depth, or height. This must be the single dimension you have the highest confidence in estimating.
The other two dimension fields for that asset MUST be set to null.
The provided dimension MUST be a number in meters, accurate to two decimal places.

EXAMPLE:
If the new supplementary assets are "iron_candelabra", "crimson_runner_rug", and "stone_gargoyle_statue":
[{"asset_id":"iron_candelabra", "width": null, "depth": null, "height": 1.40}, {"asset_id":"crimson_runner_rug", "width": 0.70, "depth": null, "height": null}, {"asset_id":"stone_gargoyle_statue", "width": null, "depth": null, "height": 0.75}]
Begin your analysis after the user input. Your output must be only the JSON array."""
    
    # Select the appropriate system instruction based on estimation_type
    if estimation_type == "supplementary_assets":
        selected_instruction = system_instruction_for_supplementary_assets
    else:
        selected_instruction = system_instruction
    
    # Build messages for any-llm
    messages = [
        {
            "role": "system",
            "content": selected_instruction
        },
        {
            "role": "user",
            "content": contents
        }
    ]
    
    # Generate content
    try:
        response = completion(
            api_key=anyllm_api_key,
            api_base=anyllm_api_base,
            provider=anyllm_provider,
            model=reasoning_model,
            reasoning_effort=reasoning_effort,
            messages=messages,
            response_format=response_schema
        )
        gc.collect()
        result = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error generating content: {e}")
        return None

    # validate schema
    try:
        response_schema.model_validate(result)
    except Exception as e:
        print(f"Error validating response schema: {e}")
        return None
    
    # validate output
    if not validate_output(concept_data, result):
        print("Error: Output does not match asset sheet")
        return None
    
    # Return as json
    return result
