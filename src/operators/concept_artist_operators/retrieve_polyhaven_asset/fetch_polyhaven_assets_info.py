link_to_polyhaven_api = {
    "hdris": "https://api.polyhaven.com/assets?t=hdris",
    "models": "https://api.polyhaven.com/assets?t=models",
    "textures": "https://api.polyhaven.com/assets?t=textures"
}


import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests


_POLYHAVEN_API_BASE = "https://api.polyhaven.com"


def _default_headers(user_agent: str = "blender-mcp") -> Dict[str, str]:
    headers = requests.utils.default_headers()
    headers.update({"User-Agent": user_agent})
    return headers


def _request_json(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    connect_timeout_seconds: float = 10.0,
    read_timeout_seconds: float = 30.0,
    max_retries: int = 5,
    retry_backoff_seconds: float = 1.0,
) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = session.get(
                url,
                params=params,
                headers=headers,
                timeout=(connect_timeout_seconds, read_timeout_seconds),
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_error = e
            if attempt >= max_retries - 1:
                break
            time.sleep(retry_backoff_seconds * (2**attempt))
    raise RuntimeError(f"Failed requesting {url}: {last_error}")


def fetch_all_assets_index(
    asset_type: str,
    *,
    session: Optional[requests.Session] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if asset_type not in {"hdris", "models", "textures"}:
        raise ValueError("asset_type must be one of: hdris, models, textures")

    own_session = session is None
    session = session or requests.Session()
    headers = headers or _default_headers()

    try:
        return _request_json(
            session,
            f"{_POLYHAVEN_API_BASE}/assets",
            params={"type": asset_type},
            headers=headers,
        )
    finally:
        if own_session:
            session.close()


def fetch_asset_info(
    asset_id: str,
    *,
    session: requests.Session,
    headers: Dict[str, str],
) -> Dict[str, Any]:
    return _request_json(session, f"{_POLYHAVEN_API_BASE}/info/{asset_id}", headers=headers)


def fetch_asset_files(
    asset_id: str,
    *,
    session: requests.Session,
    headers: Dict[str, str],
) -> Dict[str, Any]:
    return _request_json(session, f"{_POLYHAVEN_API_BASE}/files/{asset_id}", headers=headers)


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _upgrade_thumbnail_url(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return value.replace("width=256&height=256", "width=768&height=768")


def _upgrade_thumbnails_in_assets(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    for _, v in data.items():
        if isinstance(v, dict) and "thumbnail_url" in v:
            v["thumbnail_url"] = _upgrade_thumbnail_url(v.get("thumbnail_url"))
    return data


def download_all_assets_info_as_three_json(
    output_dir: str | Path,
    *,
    include_files: bool = False,
    include_full_info: bool = False,
    resume: bool = True,
    progress_every: int = 50,
    user_agent: str = "blender-mcp",
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = _default_headers(user_agent=user_agent)
    results: Dict[str, Path] = {}

    with requests.Session() as session:
        for asset_type in ("hdris", "models", "textures"):
            index = fetch_all_assets_index(asset_type, session=session, headers=headers)
            out_path = output_dir / f"polyhaven_{asset_type}.json"

            if not include_full_info and not include_files:
                _upgrade_thumbnails_in_assets(index)
                _atomic_write_json(out_path, index)
                results[asset_type] = out_path
                continue

            assets: Dict[str, Any] = {}
            if resume:
                existing = _read_json_if_exists(out_path)
                if isinstance(existing, dict):
                    assets = existing

            asset_ids = list(index.keys())
            total = len(asset_ids)
            done = 0

            for i, asset_id in enumerate(asset_ids, start=1):
                if resume and asset_id in assets:
                    done += 1
                    continue

                if include_full_info:
                    info = fetch_asset_info(asset_id, session=session, headers=headers)
                else:
                    info = index.get(asset_id, {})

                if isinstance(info, dict) and "thumbnail_url" in info:
                    info = dict(info)
                    info["thumbnail_url"] = _upgrade_thumbnail_url(info.get("thumbnail_url"))

                if include_files:
                    info = dict(info)
                    info["files"] = fetch_asset_files(asset_id, session=session, headers=headers)

                assets[asset_id] = info
                done += 1

                if progress_every > 0 and (done % progress_every == 0):
                    print(f"[{asset_type}] {done}/{total} fetched")
                    _atomic_write_json(out_path, assets)

            _atomic_write_json(out_path, assets)
            results[asset_type] = out_path

    return results


if __name__ == "__main__":
    # Writes:
    # - polyhaven_hdris.json
    # - polyhaven_models.json
    # - polyhaven_textures.json
    # into ./polyhaven_assets_info_raw
    written = download_all_assets_info_as_three_json("./polyhaven_assets_info_raw", include_files=False)
    print({k: str(v) for k, v in written.items()})