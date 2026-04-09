# Trim description for HDRIs
 
import json
import re
from pathlib import Path
from typing import Any, Dict


def trim_polyhaven_hdri_description(description: Any) -> Any:
    if not isinstance(description, str):
        return description

    s = description.strip()
    if not s:
        return s

    for _ in range(4):
        prev = s
        s = re.sub(r"^\s*(?:(?:free|unclipped)\b[\s,]*)+", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*\d+\s*k\b[\s,]*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*hdri\b[\s,]*", "", s, flags=re.IGNORECASE)
        if s == prev:
            break

    s = re.sub(r"^\s*(?:of|capturing)\b\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*[\s,:;\-\.]+\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def trim_polyhaven_texture_description(description: Any) -> Any:
    if not isinstance(description, str):
        return description

    s = description.strip()
    if not s:
        return s

    for _ in range(4):
        prev = s
        s = re.sub(r"^\s*free\b[\s,]*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*\d+\s*k\b[\s,]*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*texture\b[\s,]*", "", s, flags=re.IGNORECASE)
        if s == prev:
            break

    s = re.sub(r"^\s*(?:of)\b\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*[\s,:;\-\.]+\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def trim_polyhaven_model_description(description: Any) -> Any:
    if not isinstance(description, str):
        return description

    s = description.strip()
    if not s:
        return s

    for _ in range(4):
        prev = s
        s = re.sub(r"^\s*free\b[\s,]*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*\(?\s*cc0\s*\)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*\d+\s*k\b[\s,]*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*(?:3d\s+)?model\b[\s,]*", "", s, flags=re.IGNORECASE)
        if s == prev:
            break

    s = re.sub(r"^\s*(?:of)\b\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*[\s,:;\-\.]+\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def trim_polyhaven_asset_name(name: Any) -> Any:
    if not isinstance(name, str):
        return name
    s = name.strip()
    if not s:
        return s
    s = re.sub(r"\s+\d+$", "", s)
    return s


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path.replace(path)


def preprocess_polyhaven_hdris_json(
    input_json_path: str | Path,
    output_dir: str | Path,
) -> Path:
    input_json_path = Path(input_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at root, got {type(data)}")

    out: Dict[str, Any] = {}
    for asset_id, asset_info in data.items():
        if not isinstance(asset_info, dict):
            out[asset_id] = asset_info
            continue
        new_info = dict(asset_info)
        if "description" in new_info:
            new_info["description"] = trim_polyhaven_hdri_description(new_info.get("description"))
        if "name" in new_info:
            new_info["name"] = trim_polyhaven_asset_name(new_info.get("name"))
        out[asset_id] = new_info

    out_path = output_dir / input_json_path.name
    _atomic_write_json(out_path, out)
    return out_path


def preprocess_polyhaven_textures_json(
    input_json_path: str | Path,
    output_dir: str | Path,
) -> Path:
    input_json_path = Path(input_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at root, got {type(data)}")

    out: Dict[str, Any] = {}
    for asset_id, asset_info in data.items():
        if not isinstance(asset_info, dict):
            out[asset_id] = asset_info
            continue
        new_info = dict(asset_info)
        if "description" in new_info:
            new_info["description"] = trim_polyhaven_texture_description(new_info.get("description"))
        if "name" in new_info:
            new_info["name"] = trim_polyhaven_asset_name(new_info.get("name"))
        out[asset_id] = new_info

    out_path = output_dir / input_json_path.name
    _atomic_write_json(out_path, out)
    return out_path


def preprocess_polyhaven_models_json(
    input_json_path: str | Path,
    output_dir: str | Path,
) -> Path:
    input_json_path = Path(input_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at root, got {type(data)}")

    out: Dict[str, Any] = {}
    for asset_id, asset_info in data.items():
        if not isinstance(asset_info, dict):
            out[asset_id] = asset_info
            continue
        new_info = dict(asset_info)
        if "description" in new_info:
            new_info["description"] = trim_polyhaven_model_description(new_info.get("description"))
        if "name" in new_info:
            new_info["name"] = trim_polyhaven_asset_name(new_info.get("name"))
        out[asset_id] = new_info

    out_path = output_dir / input_json_path.name
    _atomic_write_json(out_path, out)
    return out_path


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    raw_dir = here / "polyhaven_assets_info_raw"
    out_dir = here / "polyhaven_assets_info"
    written_hdris = preprocess_polyhaven_hdris_json(raw_dir / "polyhaven_hdris.json", out_dir)
    written_models = preprocess_polyhaven_models_json(raw_dir / "polyhaven_models.json", out_dir)
    written_textures = preprocess_polyhaven_textures_json(raw_dir / "polyhaven_textures.json", out_dir)
    print(str(written_hdris))
    print(str(written_models))
    print(str(written_textures))