#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from model_downloader import (
    DEFAULT_MODEL_NAME,
    available_model_names,
    download_selected_model,
    model_weight_paths,
)
from sam_hover_mask_gui import (
    MODELS_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    SegmentAnythingOnnx,
    build_model_cache_token,
    embedding_store_path_for_image,
    save_embedding_cache_for_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively precompute SAM image embeddings for a folder tree so "
            "the click GUI can open images quickly later."
        ),
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help="Root image folder. If omitted, a folder picker is shown.",
    )
    parser.add_argument(
        "--source-folder",
        "--folder",
        dest="source_folder",
        help="Root image folder. Overrides the positional folder.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        choices=available_model_names(),
        help="Model preset to download/use.",
    )
    parser.add_argument(
        "--encoder-path",
        default="",
        help="Optional custom SAM encoder ONNX path. Must be paired with --decoder-path.",
    )
    parser.add_argument(
        "--decoder-path",
        default="",
        help="Optional custom SAM decoder ONNX path. Must be paired with --encoder-path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings even when a fresh cache file already exists.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional summary JSON path. Defaults to no summary file.",
    )
    return parser.parse_args()


def select_folder_with_gui() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("No folder was provided and tkinter is unavailable.") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update_idletasks()
    selected = filedialog.askdirectory(parent=root, title="Select root image folder")
    root.destroy()
    if not selected:
        raise RuntimeError("No folder selected.")
    return Path(selected).expanduser().resolve()


def resolve_source_folder(args: argparse.Namespace) -> Path:
    raw_folder = args.source_folder or args.folder
    if raw_folder:
        folder = Path(str(raw_folder)).expanduser().resolve()
    else:
        folder = select_folder_with_gui()

    if not folder.exists():
        raise RuntimeError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise RuntimeError(f"Path is not a folder: {folder}")
    return folder


def iter_image_paths(root_folder: Path) -> list[Path]:
    paths = [
        path.resolve()
        for path in root_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda path: str(path).lower())


def resolve_model_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    encoder_raw = str(args.encoder_path or "").strip()
    decoder_raw = str(args.decoder_path or "").strip()
    if bool(encoder_raw) ^ bool(decoder_raw):
        raise RuntimeError("Provide both --encoder-path and --decoder-path, or neither.")

    if encoder_raw and decoder_raw:
        encoder_path = Path(encoder_raw).expanduser().resolve()
        decoder_path = Path(decoder_raw).expanduser().resolve()
        if not encoder_path.exists() or not decoder_path.exists():
            raise RuntimeError(
                "Custom encoder/decoder paths do not exist: "
                f"{encoder_path} / {decoder_path}"
            )
        return encoder_path, decoder_path

    encoder_path, decoder_path = model_weight_paths(args.model_name, MODELS_DIR)
    if not encoder_path.exists() or not decoder_path.exists():
        encoder_path, decoder_path = download_selected_model(
            model_name=args.model_name,
            models_dir=MODELS_DIR,
        )
    return encoder_path, decoder_path


def load_embedding_metadata(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            metadata_json = data["metadata_json"] if "metadata_json" in data.files else None
    except Exception:
        return {}
    if metadata_json is None:
        return {}

    try:
        if np.ndim(metadata_json) == 0:
            metadata_text = str(metadata_json.item())
        else:
            metadata_text = str(np.asarray(metadata_json).reshape(-1)[0])
        payload = json.loads(metadata_text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def is_embedding_cache_fresh(image_path: Path, model_cache_token: str) -> bool:
    cache_path = embedding_store_path_for_image(image_path, model_cache_token)
    payload = load_embedding_metadata(cache_path)
    if not payload:
        return False
    if payload.get("model_cache_token") != model_cache_token:
        return False

    try:
        stat = image_path.stat()
    except Exception:
        return False

    cached_size = payload.get("image_file_size_bytes")
    cached_mtime_ns = payload.get("image_mtime_ns")
    try:
        if cached_size is not None and int(cached_size) != int(stat.st_size):
            return False
        if cached_mtime_ns is not None and int(cached_mtime_ns) != int(stat.st_mtime_ns):
            return False
    except Exception:
        return False
    return True


def load_image_rgb(image_path: Path) -> np.ndarray:
    try:
        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load image: {image_path}") from exc


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root_folder = resolve_source_folder(args)
    image_paths = iter_image_paths(root_folder)
    if not image_paths:
        raise RuntimeError(f"No supported images found under: {root_folder}")

    encoder_path, decoder_path = resolve_model_paths(args)
    model = SegmentAnythingOnnx(str(encoder_path), str(decoder_path))
    model_cache_token = build_model_cache_token(
        model_name=args.model_name,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        image_size=int(model.image_size),
    )

    started_at = utc_now_iso()
    cached = 0
    skipped = 0
    failed = 0
    records: list[dict[str, Any]] = []
    total = len(image_paths)

    print(f"Root folder: {root_folder}", flush=True)
    print(f"Images found: {total}", flush=True)
    print(f"Model: {args.model_name}", flush=True)

    for index, image_path in enumerate(image_paths, start=1):
        rel_path = image_path.relative_to(root_folder)
        cache_path = embedding_store_path_for_image(image_path, model_cache_token)

        if not args.force and is_embedding_cache_fresh(image_path, model_cache_token):
            skipped += 1
            print(f"[{index}/{total}] skip fresh: {rel_path}", flush=True)
            records.append(
                {
                    "image_path": str(image_path),
                    "relative_path": str(rel_path),
                    "status": "skipped_fresh",
                    "embedding_cache_path": str(cache_path),
                }
            )
            continue

        try:
            image_np = load_image_rgb(image_path)
            model.set_image(image_np)
            embedding = model.get_embedding()
            if embedding is None:
                raise RuntimeError("Model did not produce an image embedding.")
            saved_path = save_embedding_cache_for_image(
                image_path=image_path,
                image_np=image_np,
                embedding=embedding,
                model_cache_token=model_cache_token,
            )
            cached += 1
            print(f"[{index}/{total}] cached: {rel_path}", flush=True)
            records.append(
                {
                    "image_path": str(image_path),
                    "relative_path": str(rel_path),
                    "status": "cached",
                    "embedding_cache_path": str(saved_path),
                    "image_size_hw": [int(image_np.shape[0]), int(image_np.shape[1])],
                }
            )
        except Exception as exc:
            failed += 1
            print(f"[{index}/{total}] failed: {rel_path} ({exc})", file=sys.stderr, flush=True)
            records.append(
                {
                    "image_path": str(image_path),
                    "relative_path": str(rel_path),
                    "status": "failed",
                    "embedding_cache_path": str(cache_path),
                    "error": str(exc),
                }
            )

    summary: dict[str, Any] = {
        "action": "precompute_embeddings",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "root_folder": str(root_folder),
        "model_name": args.model_name,
        "encoder_path": str(encoder_path),
        "decoder_path": str(decoder_path),
        "model_cache_token": model_cache_token,
        "total_images": total,
        "cached_count": cached,
        "skipped_fresh_count": skipped,
        "failed_count": failed,
        "force": bool(args.force),
        "records": records,
    }

    if args.summary_json:
        summary_path = Path(str(args.summary_json)).expanduser().resolve()
        write_summary(summary_path, summary)
        print(f"Summary JSON: {summary_path}", flush=True)

    print(
        f"Done: {cached} cached, {skipped} skipped fresh, {failed} failed, {total} total.",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
