"""
Process Polyhaven assets info JSON files and generate text embeddings.

This script reads the JSON files containing asset metadata, extracts the 'name' 
and 'description' fields, generates embeddings using fastembed, and saves them
in numpy format for efficient loading. Index-to-ID mappings are saved as JSON.

Output files for each asset type (hdris, models, textures):
- {type}_name_embeddings.npy: numpy array of name embeddings
- {type}_description_embeddings.npy: numpy array of description embeddings  
- {type}_index_to_id.json: mapping from index to asset ID
- {type}_metadata.json: full metadata for each asset (keyed by ID)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from fastembed import TextEmbedding

MODEL_CACHE_DIR = str(Path(__file__).resolve().parent / "bge_small_en_v1.5_model")


# Asset type mapping: type number to string name
TYPE_MAP = {0: "hdris", 1: "textures", 2: "models"}
TYPE_NAME_TO_FILE = {
    "hdris": "polyhaven_hdris.json",
    "models": "polyhaven_models.json",
    "textures": "polyhaven_textures.json",
}


def get_assets_info_dir() -> Path:
    """Get the directory containing asset info files."""
    return Path(__file__).parent / "polyhaven_assets_info"


def get_embeddings_dir() -> Path:
    """Get the directory for storing embeddings."""
    embeddings_dir = Path(__file__).parent / "polyhaven_embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    return embeddings_dir


def load_json_file(filepath: Path) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(filepath: Path, data: Any) -> None:
    """Save data to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_asset_type(
    asset_type: str,
    assets_data: Dict[str, Any],
    embedding_model: TextEmbedding,
    embeddings_dir: Path,
) -> None:
    """
    Process assets of a specific type and generate embeddings.
    
    Args:
        asset_type: Type name (hdris, models, textures)
        assets_data: Dictionary of assets keyed by ID
        embedding_model: The fastembed TextEmbedding model
        embeddings_dir: Directory to save embeddings
    """
    print(f"Processing {asset_type}...")
    
    # Extract data in consistent order
    asset_ids: List[str] = []
    names: List[str] = []
    descriptions: List[str] = []
    metadata: Dict[str, Any] = {}
    
    for asset_id, asset_info in assets_data.items():
        asset_ids.append(asset_id)
        names.append(asset_info.get("name", ""))
        descriptions.append(asset_info.get("description", ""))
        metadata[asset_id] = asset_info
    
    print(f"  Found {len(asset_ids)} assets")
    
    # Generate embeddings for names
    print(f"  Generating name embeddings...")
    name_embeddings = list(embedding_model.embed(names))
    name_embeddings_array = np.array(name_embeddings, dtype=np.float32)
    
    # Generate embeddings for descriptions
    print(f"  Generating description embeddings...")
    description_embeddings = list(embedding_model.embed(descriptions))
    description_embeddings_array = np.array(description_embeddings, dtype=np.float32)
    
    # Save embeddings as numpy files
    name_emb_path = embeddings_dir / f"{asset_type}_name_embeddings.npy"
    desc_emb_path = embeddings_dir / f"{asset_type}_description_embeddings.npy"
    np.save(name_emb_path, name_embeddings_array)
    np.save(desc_emb_path, description_embeddings_array)
    print(f"  Saved embeddings to {name_emb_path.name} and {desc_emb_path.name}")
    
    # Save index-to-ID mapping
    index_to_id_path = embeddings_dir / f"{asset_type}_index_to_id.json"
    save_json_file(index_to_id_path, asset_ids)
    print(f"  Saved index mapping to {index_to_id_path.name}")
    
    # Save metadata
    metadata_path = embeddings_dir / f"{asset_type}_metadata.json"
    save_json_file(metadata_path, metadata)
    print(f"  Saved metadata to {metadata_path.name}")
    
    print(f"  Done processing {asset_type}")


def process_all_assets() -> None:
    """Process all asset types and generate embeddings."""
    assets_info_dir = get_assets_info_dir()
    embeddings_dir = get_embeddings_dir()
    
    print("Initializing embedding model...")
    embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=MODEL_CACHE_DIR, local_files_only=True)
    print("Model initialized successfully")
    print("-" * 50)
    
    for asset_type, filename in TYPE_NAME_TO_FILE.items():
        json_path = assets_info_dir / filename
        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping {asset_type}")
            continue
        
        assets_data = load_json_file(json_path)
        process_asset_type(asset_type, assets_data, embedding_model, embeddings_dir)
        print("-" * 50)
    
    print("All assets processed successfully!")


if __name__ == "__main__":
    process_all_assets()
