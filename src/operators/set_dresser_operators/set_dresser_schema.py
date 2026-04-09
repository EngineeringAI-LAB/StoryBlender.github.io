"""Schema definitions for supplementary asset generation by Set Dresser."""

from pydantic import BaseModel
from typing import Optional, List, Literal


class SupplementaryAsset(BaseModel):
    """Represents a single decorative/supplementary asset without text_to_image_prompt."""
    asset_id: str
    description: str
    is_reused: bool = False


class SupplementaryAssetWithPrompt(BaseModel):
    """Represents a single decorative/supplementary asset with text_to_image_prompt."""
    asset_id: str
    description: str
    text_to_image_prompt: str
    is_reused: bool = False


class SupplementaryAssetTextToImagePrompt(BaseModel):
    """Text-to-image prompt for a supplementary asset."""
    asset_id: str
    text_to_image_prompt: str


class SupplementaryLayoutDescription(BaseModel):
    """Layout description for supplementary assets using existing assets as anchors."""
    asset_id: str
    anchor_asset_id: str
    relationship: Literal['on_top_of', 'on_the_left_of', 'on_the_right_of', 'in_front_of', 'behind']
    distance: float
    description: str


class SupplementarySceneSetup(BaseModel):
    """Scene setup containing only supplementary assets and their layout descriptions."""
    asset_ids: List[str]
    layout_descriptions: List[SupplementaryLayoutDescription]


class SupplementarySceneDetail(BaseModel):
    """Scene detail for supplementary assets."""
    scene_id: int
    scene_setup: SupplementarySceneSetup


# --- Output Schema for Single Scene Generation ---

class SupplementaryAssetSheetForScene(BaseModel):
    """New supplementary assets generated for a single scene."""
    assets: List[SupplementaryAsset]


class SupplementarySceneOutput(BaseModel):
    """Output schema for a single scene's supplementary asset generation (without prompts)."""
    scene_id: int
    asset_sheet: SupplementaryAssetSheetForScene
    scene_detail: SupplementarySceneSetup


class SupplementaryTextToImagePromptSheet(BaseModel):
    """List of text_to_image_prompts for supplementary assets."""
    prompts: List[SupplementaryAssetTextToImagePrompt]


# --- All Scenes Output Schema (for single LLM call) ---

class AllScenesSupplementaryOutput(BaseModel):
    """Output schema for generating supplementary assets for all scenes in one call.
    
    When a scene has a reference_scene_id, assets are reused from the reference scene
    (marked with is_reused=True). New assets have is_reused=False.
    """
    scenes: List[SupplementarySceneOutput]


# --- Full Output Schema (after merging all scenes) ---

class SupplementaryAssetsOutput(BaseModel):
    """Complete output containing all supplementary assets and scene details."""
    asset_sheet: List[SupplementaryAssetWithPrompt]
    scene_details: List[SupplementarySceneDetail]


if __name__ == '__main__':
    # Example usage for validation
    example_scene_output = {
        "scene_id": 1,
        "asset_sheet": {
            "assets": [
                {
                    "asset_id": "wall_torch_1",
                    "description": "A medieval wall-mounted torch with iron bracket"
                }
            ]
        },
        "scene_detail": {
            "asset_ids": ["wall_torch_1"],
            "layout_descriptions": [
                {
                    "asset_id": "wall_torch_1",
                    "anchor_asset_id": "magic_mirror",
                    "relationship": "on_the_left_of",
                    "distance": 1.5,
                    "description": "Mounted on the wall 1.5 meters to the left of the magic mirror"
                }
            ]
        }
    }

    try:
        validated = SupplementarySceneOutput(**example_scene_output)
        print("Schema validation successful!")
    except Exception as e:
        print(f"Schema validation failed: {e}")
