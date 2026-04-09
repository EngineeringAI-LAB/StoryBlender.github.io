from pydantic import BaseModel
from typing import Optional, List, Literal

# Asset
AssetType = Literal['character', 'object']

# Type definitions for camera properties
Angle = Literal['eye-level', 'high angle', 'low angle']
Distance = Literal['close-up', 'medium shot', 'long shot']
Movement = Literal['static', 'pan', 'orbit', 'zoom in', 'zoom out']
Direction = Literal['left', 'right', 'up', 'down']
Mode = Literal['perspective', 'orthographic']
LookAt = Literal['lock', 'track']

# Scene
SceneType = Literal['indoor', 'outdoor']

# Shot
ModificationType = Literal['add', 'remove', 'transform']

# --- Component Models ---

class ShotOutline(BaseModel):
    """Represents a single shot within the high-level storyboard outline."""
    shot_id: int
    shot_description: str

class SceneOutline(BaseModel):
    """Represents a single scene within the high-level storyboard outline."""
    scene_id: int
    scene_description: str
    shots: List[ShotOutline]

class AssetWithoutPrompt(BaseModel):
    """Represents a single unique character or object in the story (without text_to_image_prompt)."""
    asset_id: str
    asset_type: AssetType
    description: str
    reference_character: Optional[str] = None

class Asset(BaseModel):
    """Represents a single unique character or object in the story (full version with text_to_image_prompt)."""
    asset_id: str
    asset_type: AssetType
    description: str
    reference_character: Optional[str] = None
    text_to_image_prompt: str

class AssetTextToImagePrompt(BaseModel):
    """Represents the text_to_image_prompt for a single asset."""
    asset_id: str
    text_to_image_prompt: str

class SceneSetup(BaseModel):
    """Describes the initial setup of a scene, including assets, layout, lighting and ground."""
    reference_scene_id: Optional[int] = None
    asset_ids: List[str]
    scene_type: SceneType
    layout_description: str
    lighting_description: str
    ground_description: str
    wall_description: Optional[str] = None

class SceneDetail(BaseModel):
    """Contains the detailed scene setup for a specific scene."""
    scene_id: int
    scene_setup: SceneSetup

class AssetModification(BaseModel):
    """Represents a change to an asset's state within a shot."""
    asset_id: str
    modification_type: ModificationType
    description: Optional[str] = None

class CharacterAction(BaseModel):
    """Represents a change to a character's action within a shot."""
    asset_id: str
    action_description: str

class CameraInstruction(BaseModel):
    """Defines the camera's properties and behavior for a single shot."""
    focus_on_ids: List[str]
    angle: Angle
    distance: Distance
    movement: Movement
    direction: Optional[Direction] = None
    description: str

class LightingModification(BaseModel):
    """Describes the lighting of a scene."""
    new_lighting_description: Optional[str] = None

class ShotDetail(BaseModel):
    """Contains all detailed information for a single shot, including modifications and camera work."""
    scene_id: int
    shot_id: int
    asset_modifications: Optional[List[AssetModification]] = None
    character_actions: Optional[List[CharacterAction]] = None
    lighting_modification: Optional[LightingModification] = None
    sound_effect: Optional[str] = None
    camera_instruction: CameraInstruction

# --- Main Storyboard Schema ---

class StoryboardWithoutPrompts(BaseModel):
    """Storyboard without text_to_image_prompt for each asset (used in first generation step)."""
    story_summary: str
    storyboard_outline: List[SceneOutline]
    asset_sheet: List[AssetWithoutPrompt]
    scene_details: List[SceneDetail]
    shot_details: List[ShotDetail]

class TextToImagePromptSheet(BaseModel):
    """A list of text_to_image_prompt for each asset (used in second generation step)."""
    prompts: List[AssetTextToImagePrompt]

class Storyboard(BaseModel):
    """The complete, top-level JSON object for the generated storyboard (full version)."""
    story_summary: str
    storyboard_outline: List[SceneOutline]
    asset_sheet: List[Asset]
    scene_details: List[SceneDetail]
    shot_details: List[ShotDetail]

# Example Usage (for demonstration)
if __name__ == '__main__':
    # This is a sample JSON object that would validate against the schema.
    # The Storyboard Agent would generate a structure like this.
    example_data = {
        "story_summary": "A short story about a person entering a cafe on a sunny afternoon.",
        "storyboard_outline": [
            {
                "scene_id": 1,
                "scene_description": "A person walks into a cafe.",
                "shots": [
                    {"shot_id": 1, "shot_description": "The person is outside the cafe door."},
                    {"shot_id": 2, "shot_description": "The person opens the door and enters."}
                ]
            }
        ],
        "asset_sheet": [
            {"asset_id": "main_character", "asset_type": "character", "description": "A person wearing a trench coat.", "text_to_image_prompt": "A person wearing a trench coat."},
            {"asset_id": "cafe_building", "asset_type": "object", "description": "A small, cozy-looking cafe with a glass door.", "text_to_image_prompt": "A small, cozy-looking cafe with a glass door."}
        ],
        "scene_details": [
            {
                "scene_id": 1,
                "scene_setup": {
                    "asset_ids": ["main_character", "cafe_building"],
                    "scene_type": "outdoor",
                    "layout_description": "The main_character stands on the sidewalk, facing the entrance of the cafe_building.",
                    "ground_description": "Concrete sidewalk",
                    "lighting_description": "Bright sunny afternoon",
                    "reference_scene_id": None
                }
            }
        ],
        "shot_details": [
            {
                "scene_id": 1,
                "shot_id": 1,
                "asset_modifications": None,
                "character_actions": None,
                "lighting_modification": None,
                "sound_effect": "Distant city traffic",
                "camera_instruction": {
                    "focus_on_ids": ["main_character", "cafe_building"],
                    "angle": "eye-level",
                    "distance": "long shot",
                    "movement": "static",
                    "direction": None,
                    "description": "A wide scene showing the character and the cafe entrance."
                }
            },
            {
                "scene_id": 1,
                "shot_id": 2,
                "asset_modifications": [
                    {"asset_id": "main_character", "modification_type": "transform", "description": "walks forward towards the cafe_building"}
                ],
                "lighting_modification": {
                    "new_lighting": True,
                    "new_lighting_description": "Dark inside the cafe"
                },
                "sound_effect": "Door bell jingles",
                "camera_instruction": {
                    "focus_on_ids": ["main_character"],
                    "angle": "eye-level",
                    "distance": "medium shot",
                    "movement": "pan",
                    "direction": None,
                    "description": "The camera follows the character as they move towards the door."
                }
            }
        ]
    }

    # Validate the example data
    try:
        storyboard_instance = Storyboard(**example_data)
        print("Schema validation successful!")
        # print(storyboard_instance.model_dump_json(indent=2))
    except Exception as e:
        print(f"Schema validation failed: {e}")
