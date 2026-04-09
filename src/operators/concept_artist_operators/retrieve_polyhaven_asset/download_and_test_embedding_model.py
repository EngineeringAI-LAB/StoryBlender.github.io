"""
Download the BAAI/bge-small-en-v1.5 model from HuggingFace to a local directory,
so it does not need to be downloaded when used for the first time.

The HuggingFace Hub cache uses symlinks (snapshots -> blobs). This script
resolves all symlinks into real files so the directory can be safely copied
(e.g. into a Blender extension) without breaking.

Usage:
    python download_and_test_embedding_model.py
"""

import shutil
from pathlib import Path
from fastembed import TextEmbedding

# Download destination: a "bge_small_en_v1.5_model" folder next to this script
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = SCRIPT_DIR / "bge_small_en_v1.5_model"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def resolve_symlinks(directory: Path):
    """Replace all symlinks under `directory` with copies of their targets."""
    count = 0
    for path in directory.rglob("*"):
        if path.is_symlink():
            target = path.resolve()
            path.unlink()
            if target.is_dir():
                shutil.copytree(target, path)
            else:
                shutil.copy2(target, path)
            count += 1
    print(f"Resolved {count} symlink(s) into real files.")


def download_model():
    """Download the quantized ONNX model via fastembed to a local cache directory."""
    print(f"Downloading {MODEL_NAME} to {MODEL_CACHE_DIR} ...")
    # Initializing TextEmbedding with cache_dir triggers the download
    emb = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODEL_CACHE_DIR), local_files_only=True)
    print(f"Model downloaded successfully to: {MODEL_CACHE_DIR}")

    # Resolve symlinks so the directory is safe to copy/distribute
    resolve_symlinks(MODEL_CACHE_DIR)
    return emb


def test_model(emb: TextEmbedding = None):
    """Load the model from the local directory with fastembed and run a test query."""
    if emb is None:
        print(f"\nLoading model from local cache: {MODEL_CACHE_DIR}")
        emb = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODEL_CACHE_DIR) , local_files_only=True)

    print("Running test embedding...")
    result = list(emb.embed("single query"))
    print(f"Embedding shape: ({len(result)}, {len(result[0])})")
    print(f"First 5 values: {result[0][:5]}")
    print("\nModel is working correctly!")


if __name__ == "__main__":
    emb = download_model()
    test_model(emb)
