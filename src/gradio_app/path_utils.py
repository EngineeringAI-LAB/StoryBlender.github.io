"""Utility functions for converting between absolute and relative paths in project JSON files.

When saving JSON files, absolute paths under project_dir are converted to portable
relative paths (e.g., "animated_models/file.glb"). When loading, relative paths are
expanded back to absolute paths using the current project_dir.

This enables seamless project transfer between different computers and platforms.
"""

import os
import re
import logging
import warnings

logger = logging.getLogger(__name__)

# Known project subdirectory names. Paths containing these folder names
# (directly under project_dir) are candidates for conversion.
# Sorted by length descending so longer names match first (e.g.,
# "resized_supplementary_assets" before "supplementary_assets").
PROJECT_SUBFOLDERS = sorted([
    "animated_models",
    "camera_blocking",
    "director",
    "formatted_model",
    "formatted_supplementary_assets",
    "layout_script",
    "models",
    "renders",
    "resized_model",
    "resized_supplementary_assets",
    "rigged_models",
    "supplementary_assets",
    "supplementary_layout_script",
], key=len, reverse=True)


def _normalize_sep(path):
    """Normalize path separators to forward slashes for cross-platform comparison."""
    return path.replace("\\", "/")


def _is_url(value):
    """Check if a string looks like a URL (should not be path-converted)."""
    return bool(re.match(r'^https?://', value, re.IGNORECASE))


def _to_relative(value, project_dir):
    """Convert a single absolute path string to a relative path under project_dir.

    Returns the original string unchanged if it is not under project_dir or
    does not contain a known project subfolder.
    """
    if not value or _is_url(value):
        return value

    norm_value = _normalize_sep(value)
    norm_project = _normalize_sep(project_dir).rstrip("/")

    # Case 1: path starts with current project_dir
    if norm_value.startswith(norm_project + "/"):
        relative = norm_value[len(norm_project) + 1:]
        # Verify it starts with a known subfolder
        for subfolder in PROJECT_SUBFOLDERS:
            if relative.startswith(subfolder + "/") or relative == subfolder:
                return relative  # store with forward slashes (portable)
        # Path is under project_dir but not in a known subfolder – leave as-is
        return value

    # Case 2: path is an old absolute path from another machine.
    # Find the *last* occurrence of a known subfolder preceded by a separator.
    for subfolder in PROJECT_SUBFOLDERS:
        pattern = "/" + subfolder + "/"
        idx = norm_value.rfind(pattern)
        if idx >= 0:
            relative = norm_value[idx + 1:]  # e.g. "animated_models/file.glb"
            return relative

    return value


def _to_absolute(value, project_dir):
    """Convert a relative path (or old absolute path) to an absolute path using project_dir.

    Returns the original string unchanged if it is already correct or not a
    project-related path.
    """
    if not value or _is_url(value):
        return value

    norm_value = _normalize_sep(value)
    norm_project = _normalize_sep(project_dir).rstrip("/")

    # Already has the correct project_dir prefix – return with native separators
    if norm_value.startswith(norm_project + "/"):
        return os.path.normpath(value)

    # Case 1: relative path starting with a known subfolder
    for subfolder in PROJECT_SUBFOLDERS:
        if norm_value.startswith(subfolder + "/") or norm_value == subfolder:
            parts = norm_value.split("/")
            return os.path.join(project_dir, *parts)

    # Case 2: old absolute path from a different machine / project_dir
    for subfolder in PROJECT_SUBFOLDERS:
        pattern = "/" + subfolder + "/"
        idx = norm_value.rfind(pattern)
        if idx >= 0:
            relative = norm_value[idx + 1:]  # e.g. "animated_models/file.glb"
            parts = relative.split("/")
            return os.path.join(project_dir, *parts)

    return value


def _walk_json(data, processor, path="$"):
    """Recursively walk a JSON-compatible structure and apply *processor* to every string value.

    Args:
        data: JSON-compatible object (dict, list, str, int, float, bool, None).
        processor: Callable that takes (string_value) and returns a new string.
        path: JSON-path string used for warning messages (internal).

    Returns:
        A new structure with processed strings.
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            try:
                result[key] = _walk_json(value, processor, path=f"{path}.{key}")
            except Exception as e:
                logger.warning("path_utils: failed to process key '%s' at %s: %s", key, path, e)
                result[key] = value
        return result
    elif isinstance(data, list):
        result = []
        for idx, item in enumerate(data):
            try:
                result.append(_walk_json(item, processor, path=f"{path}[{idx}]"))
            except Exception as e:
                logger.warning("path_utils: failed to process item at %s[%d]: %s", path, idx, e)
                result.append(item)
        return result
    elif isinstance(data, str):
        try:
            return processor(data)
        except Exception as e:
            logger.warning("path_utils: failed to process string at %s: %s", path, e)
            return data
    else:
        return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_paths_relative(data, project_dir):
    """Convert all absolute project paths in *data* to portable relative paths.

    Call this **before** writing JSON to disk so the file is transferable.

    Args:
        data: Parsed JSON data (dict or list).
        project_dir: Current project directory (absolute path).

    Returns:
        A new data structure with converted paths. The original is not mutated.
    """
    if not project_dir:
        return data
    try:
        return _walk_json(data, lambda v: _to_relative(v, project_dir))
    except Exception as e:
        logger.warning("path_utils: make_paths_relative failed, returning data unchanged: %s", e)
        return data


def make_paths_absolute(data, project_dir):
    """Convert all relative (or foreign absolute) project paths in *data* to
    absolute paths using the current *project_dir*.

    Call this **after** reading JSON from disk so downstream code receives
    correct absolute paths for the current machine.

    Args:
        data: Parsed JSON data (dict or list).
        project_dir: Current project directory (absolute path).

    Returns:
        A new data structure with converted paths. The original is not mutated.
    """
    if not project_dir:
        return data
    try:
        return _walk_json(data, lambda v: _to_absolute(v, project_dir))
    except Exception as e:
        logger.warning("path_utils: make_paths_absolute failed, returning data unchanged: %s", e)
        return data


def load_json_with_paths(filepath, project_dir):
    """Load a JSON file and convert all project paths to absolute.

    Convenience wrapper around json.load + make_paths_absolute.

    Args:
        filepath: Path to the JSON file.
        project_dir: Current project directory.

    Returns:
        Parsed JSON data with absolute paths.

    Raises:
        FileNotFoundError, json.JSONDecodeError on failure.
    """
    import json
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return make_paths_absolute(data, project_dir)


def save_json_with_paths(data, filepath, project_dir, indent=2):
    """Convert all project paths to relative and write JSON to disk.

    Convenience wrapper around make_paths_relative + json.dump.

    Args:
        data: JSON-serializable data.
        filepath: Destination path.
        project_dir: Current project directory.
        indent: JSON indentation (default 2).
    """
    import json
    portable_data = make_paths_relative(data, project_dir)
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(portable_data, f, indent=indent)
