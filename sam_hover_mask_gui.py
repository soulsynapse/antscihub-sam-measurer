from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from tkinter import filedialog, messagebox
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np
from PIL import Image, ImageTk
import sam_mask_engine as mask_engine
from model_downloader import (
    DEFAULT_MODEL_NAME,
    available_model_names,
    download_selected_model,
    model_weight_paths,
)

STAGE_DIR = Path(__file__).resolve().parent
MODELS_DIR = STAGE_DIR / "models"
ANNOTATION_FILE_SUFFIX = ".sam_clicks.npz"
ANNOTATION_METADATA_FILE_SUFFIX = ".sam_clicks.json"
EMBEDDING_FILE_SUFFIX = ".sam_embedding.npz"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_OVERLAY_ALPHA = 0.62
HOVER_MASK_OVERLAY_ALPHA = 0.45
IGNORE_MASK_OVERLAY_ALPHA = 0.58
IGNORE_MASK_OVERLAY_COLOR = (255, 35, 35)
IGNORE_MASK_LABEL_COLOR = "#ff2323"
MASK_OVERLAY_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 220, 255),
    (255, 190, 0),
    (255, 80, 200),
    (90, 230, 80),
    (255, 120, 50),
    (160, 150, 255),
    (255, 70, 90),
    (50, 210, 150),
)
OUTLINE_COLOR_CHOICES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("Neon cyan", (0, 255, 255)),
    ("Neon lime", (80, 255, 0)),
    ("Neon pink", (255, 45, 210)),
    ("Neon yellow", (255, 255, 0)),
    ("Neon orange", (255, 120, 0)),
    ("Neon violet", (180, 80, 255)),
)
OUTLINE_COLOR_MAP = dict(OUTLINE_COLOR_CHOICES)
DEFAULT_OUTLINE_COLOR_NAME = OUTLINE_COLOR_CHOICES[0][0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM hover mask stage entrypoint (runner + direct GUI modes).",
    )
    parser.add_argument("--source-input")
    parser.add_argument("--output-result-json")
    parser.add_argument("--params-json")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Force GUI mode even when runner args are present.",
    )
    parser.add_argument(
        "--gui-source-input",
        default="",
        help="Optional source input path when launching GUI directly.",
    )
    parser.add_argument(
        "--gui-params-json",
        default="",
        help="Optional params JSON to pre-load when launching GUI directly.",
    )
    return parser.parse_args()


def has_text(raw: str | None) -> bool:
    return isinstance(raw, str) and bool(raw.strip())


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON file must contain an object: {path}")
    return payload


def load_json_object_optional(raw_path: str) -> dict[str, Any]:
    path_text = str(raw_path or "").strip()
    if not path_text:
        return {}
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Params JSON does not exist: {path}")
    return load_json_object(path)


def parse_bool_like(raw: Any) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return None


def as_bool(params: dict[str, Any], key: str, default: bool) -> bool:
    parsed = parse_bool_like(params.get(key, default))
    if parsed is None:
        return bool(default)
    return parsed


def parse_float_like(raw: Any) -> float | None:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            return None
    return None


def resolve_run_mode() -> str:
    run_mode = os.environ.get("PIPEYARD_RUN_MODE", "").strip().lower()
    if run_mode in {"visual", "headless"}:
        return run_mode

    if parse_bool_like(os.environ.get("PIPEYARD_VISUAL_MODE")) is True:
        return "visual"
    if parse_bool_like(os.environ.get("PIPEYARD_HEADLESS")) is True:
        return "headless"
    return "headless"


def resolve_preload_source_input(params: dict[str, Any]) -> Path | None:
    for key in ("source_input", "image_path", "gui_source_input"):
        raw = params.get(key)
        if isinstance(raw, str) and raw.strip():
            return Path(raw).expanduser().resolve()
    return None


def _path_size_for_cache_token(path: Path) -> str:
    try:
        return str(int(path.stat().st_size))
    except Exception:
        return "unknown"


def build_legacy_model_cache_token(
    model_name: str,
    encoder_path: Path,
    decoder_path: Path,
    image_size: int,
) -> str:
    return "|".join(
        [
            str(model_name).strip() or DEFAULT_MODEL_NAME,
            str(encoder_path.resolve()),
            str(decoder_path.resolve()),
            str(int(image_size)),
        ]
    )


def build_model_cache_token(
    model_name: str,
    encoder_path: Path,
    decoder_path: Path,
    image_size: int,
) -> str:
    encoder = Path(encoder_path)
    decoder = Path(decoder_path)
    return "|".join(
        [
            "sam_embedding_cache_v2",
            str(model_name).strip() or DEFAULT_MODEL_NAME,
            encoder.name,
            _path_size_for_cache_token(encoder),
            decoder.name,
            _path_size_for_cache_token(decoder),
            str(int(image_size)),
        ]
    )


def build_model_cache_metadata(
    model_name: str,
    encoder_path: Path,
    decoder_path: Path,
    image_size: int,
) -> dict[str, Any]:
    encoder = Path(encoder_path)
    decoder = Path(decoder_path)
    return {
        "model_cache_token_version": 2,
        "model_name": str(model_name).strip() or DEFAULT_MODEL_NAME,
        "model_image_size": int(image_size),
        "encoder_name": encoder.name,
        "decoder_name": decoder.name,
        "encoder_size_bytes": _path_size_for_cache_token(encoder),
        "decoder_size_bytes": _path_size_for_cache_token(decoder),
        "legacy_model_cache_token": build_legacy_model_cache_token(
            model_name=model_name,
            encoder_path=encoder,
            decoder_path=decoder,
            image_size=image_size,
        ),
    }


def embedding_store_path_for_image(image_path: Path, model_cache_token: str) -> Path:
    token_hash = hashlib.sha1(model_cache_token.encode("utf-8")).hexdigest()[:12]
    return image_path.with_name(f"{image_path.name}.{token_hash}{EMBEDDING_FILE_SUFFIX}")


def embedding_cache_candidate_paths_for_image(
    image_path: Path,
    model_cache_token: str,
) -> list[Path]:
    exact_path = embedding_store_path_for_image(image_path, model_cache_token)
    prefix = f"{image_path.name}."
    candidates: list[Path] = [exact_path]
    try:
        for sibling in sorted(image_path.parent.iterdir(), key=lambda path: path.name.lower()):
            if (
                sibling.is_file()
                and sibling.name.startswith(prefix)
                and sibling.name.endswith(EMBEDDING_FILE_SUFFIX)
            ):
                candidates.append(sibling)
    except Exception:
        pass

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def embedding_cache_payload_matches_image(
    payload: dict[str, Any],
    image_path: Path,
    image_shape_hw: tuple[int, int] | None = None,
) -> bool:
    image_hw = payload.get("image_size_hw")
    if image_shape_hw is not None:
        if not (isinstance(image_hw, list) and len(image_hw) == 2):
            return False
        if (int(image_hw[0]), int(image_hw[1])) != image_shape_hw:
            return False

    cached_size = payload.get("image_file_size_bytes")
    if cached_size is not None:
        try:
            if int(cached_size) != int(image_path.stat().st_size):
                return False
        except Exception:
            return False
    return True


def embedding_cache_payload_matches_model(
    payload: dict[str, Any],
    model_cache_token: str,
    model_metadata: dict[str, Any],
    allow_legacy_unknown_token: bool = False,
) -> bool:
    if payload.get("model_cache_token") == model_cache_token:
        return True

    comparable_keys = (
        "model_name",
        "model_image_size",
        "encoder_name",
        "decoder_name",
        "encoder_size_bytes",
        "decoder_size_bytes",
    )
    if all(key in payload for key in comparable_keys):
        try:
            return all(
                str(payload.get(key)) == str(model_metadata.get(key))
                for key in comparable_keys
            )
        except Exception:
            return False

    return bool(
        allow_legacy_unknown_token and isinstance(payload.get("model_cache_token"), str)
    )


def annotation_store_path_for_image(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.name}{ANNOTATION_FILE_SUFFIX}")


def annotation_metadata_path_for_image(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.name}{ANNOTATION_METADATA_FILE_SUFFIX}")


def save_embedding_cache_for_image(
    image_path: Path,
    image_np: np.ndarray,
    embedding: np.ndarray,
    model_cache_token: str,
    model_metadata: dict[str, Any] | None = None,
) -> Path:
    cache_path = embedding_store_path_for_image(
        image_path=image_path,
        model_cache_token=model_cache_token,
    )
    h, w = image_np.shape[:2]
    image_file_size = None
    image_mtime_ns = None
    try:
        stat = image_path.stat()
        image_file_size = int(stat.st_size)
        image_mtime_ns = int(stat.st_mtime_ns)
    except Exception:
        image_file_size = None
        image_mtime_ns = None

    payload: dict[str, Any] = {
        "version": 1,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "model_cache_token": model_cache_token,
        "image_path": str(image_path),
        "image_name": image_path.name,
        "image_size_hw": [int(h), int(w)],
        "image_file_size_bytes": image_file_size,
        "image_mtime_ns": image_mtime_ns,
    }
    payload.update(dict(model_metadata or {}))

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                embedding=np.asarray(embedding, dtype=np.float32),
                metadata_json=np.asarray(json.dumps(payload), dtype=np.str_),
            )
        tmp_path.replace(cache_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to write embedding cache: {cache_path}") from exc

    return cache_path


def count_saved_masks(annotation_path: Path) -> int:
    if not annotation_path.exists():
        return 0
    try:
        with np.load(annotation_path, allow_pickle=False) as data:
            if "masks_packed" in data.files:
                masks_packed = np.asarray(data["masks_packed"])
                if masks_packed.ndim != 3:
                    return 0
                return int(masks_packed.shape[0])
            if "masks" in data.files:
                masks = np.asarray(data["masks"])
                if masks.ndim == 0:
                    return 0
                return int(masks.shape[0])
    except Exception:
        return 0
    return 0


def load_json_object_from_path_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    if isinstance(loaded, dict):
        return loaded
    return {}


class SegmentAnythingOnnx:
    """Minimal standalone SAM ONNX wrapper for point-prompt mask prediction."""

    def __init__(self, encoder_path: str, decoder_path: str, image_size: int = 1024):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "onnxruntime is required. Install it with: pip install onnxruntime"
            ) from exc

        self.image_size = int(image_size)
        self.encoder: Any = ort.InferenceSession(str(encoder_path))
        self.decoder: Any = ort.InferenceSession(str(decoder_path))
        self._image: np.ndarray | None = None
        self._embedding: np.ndarray | None = None

    def set_image(self, image_rgb: np.ndarray) -> None:
        self._image = np.asarray(image_rgb, dtype=np.uint8)
        self._embedding = self._compute_image_embedding(self._image)

    def set_image_with_embedding(
        self, image_rgb: np.ndarray, embedding: np.ndarray
    ) -> None:
        self._image = np.asarray(image_rgb, dtype=np.uint8)
        self._embedding = np.asarray(embedding, dtype=np.float32)

    def get_embedding(self) -> np.ndarray | None:
        if self._embedding is None:
            return None
        return np.asarray(self._embedding)

    def predict_mask_from_points(
        self, points: np.ndarray, point_labels: np.ndarray
    ) -> np.ndarray:
        if self._image is None or self._embedding is None:
            raise RuntimeError("Call set_image before prediction.")

        image = self._image
        input_point = np.asarray(points, dtype=np.float32)
        input_label = np.asarray(point_labels, dtype=np.int32)

        # SAM-style sentinel point/label.
        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]], dtype=np.float32)], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1], dtype=np.int32)], axis=0)[None, :].astype(np.float32)

        scale, new_h, new_w = self._compute_resize_scale(image)
        _ = scale
        onnx_coord = (
            onnx_coord.astype(np.float32)
            * np.array([new_w / image.shape[1], new_h / image.shape[0]], dtype=np.float32)
        )

        input_names = [x.name for x in self.decoder.get_inputs()]
        if len(input_names) <= 3:
            outputs = self.decoder.run(
                None,
                {
                    "image_embeddings": self._embedding,
                    "point_coords": onnx_coord,
                    "point_labels": onnx_label,
                },
            )
            scores, masks = outputs
            _ = scores
            mask = self._postprocess_masks(
                masks, self.image_size, new_h, new_w, image.shape[:2]
            )[0, 0]
        else:
            decoder_inputs = {
                "image_embeddings": self._embedding,
                "point_coords": onnx_coord,
                "point_labels": onnx_label,
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                "has_mask_input": np.array([-1], dtype=np.float32),
                "orig_im_size": np.array(image.shape[:2], dtype=np.float32),
            }
            outputs = self.decoder.run(None, decoder_inputs)
            mask = outputs[0][0, 0]

        return mask.astype(np.float32)

    def _compute_resize_scale(self, image: np.ndarray) -> tuple[float, int, int]:
        h, w = image.shape[:2]
        if w > h:
            scale = self.image_size / float(w)
            new_w = self.image_size
            new_h = int(round(h * scale))
        else:
            scale = self.image_size / float(h)
            new_h = self.image_size
            new_w = int(round(w * scale))
        return float(scale), int(new_h), int(new_w)

    def _compute_image_embedding(self, image: np.ndarray) -> np.ndarray:
        scale, new_h, new_w = self._compute_resize_scale(image)
        _ = scale
        resized = Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BILINEAR)
        x = np.asarray(resized).astype(np.float32)
        x = (x - np.array([123.675, 116.28, 103.53], dtype=np.float32)) / np.array(
            [58.395, 57.12, 57.375], dtype=np.float32
        )
        x = np.pad(
            x,
            ((0, self.image_size - x.shape[0]), (0, self.image_size - x.shape[1]), (0, 0)),
        )
        x = x.transpose(2, 0, 1)[None, :, :, :]

        in_name = self.encoder.get_inputs()[0].name
        if in_name == "image":
            output = self.encoder.run(None, {"image": x})
        else:
            output = self.encoder.run(None, {"x": x})
        return output[0]

    @staticmethod
    def _postprocess_masks(
        masks: np.ndarray, img_size: int, input_h: int, input_w: int, original_hw: tuple[int, int]
    ) -> np.ndarray:
        # Follow the same general postprocess pattern as SAM ONNX wrappers.
        m = masks.squeeze(0).transpose(1, 2, 0)
        m = np.asarray(Image.fromarray(m[..., 0]).resize((img_size, img_size), Image.Resampling.BILINEAR))[..., None]
        m = m[:input_h, :input_w, :]
        out_h, out_w = original_hw
        m = np.asarray(Image.fromarray(m[..., 0]).resize((out_w, out_h), Image.Resampling.BILINEAR))[..., None]
        m = m.transpose(2, 0, 1)[None, :, :, :]
        return m


class SamHoverMaskApp:
    def __init__(
        self,
        root: tk.Tk,
        initial_source_input: Path | None = None,
        initial_params: dict[str, Any] | None = None,
    ) -> None:
        self.root = root
        self.root.title("SAM Hover Mask")
        self.root.geometry("1280x860")

        self.model: SegmentAnythingOnnx | None = None
        self.image_path: Path | None = None
        self.image_pil: Image.Image | None = None
        self.image_np: np.ndarray | None = None  # RGB original
        self.display_base_np: np.ndarray | None = None  # RGB resized for GUI
        self.display_photo: ImageTk.PhotoImage | None = None
        self.display_size: tuple[int, int] = (1, 1)
        self.zoom_factor = 1.0
        self._zoom_min = 0.2
        self._zoom_max = 8.0
        self._erase_radius_display_px = 16.0
        self.view_x = 0.0
        self.view_y = 0.0
        self._middle_pan_last_canvas_xy: tuple[float, float] | None = None

        self.encoder_var = tk.StringVar()
        self.decoder_var = tk.StringVar()
        self.model_name_var = tk.StringVar(value=DEFAULT_MODEL_NAME)
        self.mask_threshold_var = tk.DoubleVar(value=0.0)
        self.mask_threshold_text_var = tk.StringVar(value="0.00")
        self.continuous_mask_only_var = tk.BooleanVar(value=True)
        self.selection_outline_var = tk.BooleanVar(value=True)
        self.selection_outline_color_var = tk.StringVar(value=DEFAULT_OUTLINE_COLOR_NAME)
        self.status_var = tk.StringVar(value="Ready. Load an image.")

        self.pos_points: list[tuple[float, float]] = []
        self.neg_points: list[tuple[float, float]] = []
        self.hover_point: tuple[float, float] | None = None
        self._hover_preview_mask: np.ndarray | None = None

        self.last_mask: np.ndarray | None = None
        self.committed_masks: list[np.ndarray] = []
        self.ignore_masks: list[np.ndarray] = []
        self._ignore_mask_metadata: list[dict[str, Any]] = []
        self._committed_overlay_signature: tuple[Any, ...] | None = None
        self._committed_overlay_rgb: np.ndarray | None = None
        self._committed_overlay_base_weight: np.ndarray | None = None
        self._seed_mask_cache: dict[tuple[int, int, bool], np.ndarray] = {}
        self._active_session_id = 0
        self._auto_advance_session_id: int | None = None
        self._pending_navigation_delta: int | None = None
        self._session_to_mask_index: dict[int, int] = {}
        self._session_metadata: dict[int, dict[str, Any]] = {}
        self._predict_lock = threading.Lock()
        self._predict_busy = False
        self._dirty_preview = False
        self._request_id = 0
        self._loaded_model_signature: tuple[str, str] | None = None
        self._folder_image_paths: list[Path] = []
        self._current_image_index = -1
        self._batch_cache_busy = False
        self.prev_image_btn: tk.Button | None = None
        self.next_image_btn: tk.Button | None = None
        self.cache_all_btn: tk.Button | None = None
        self.mask_threshold_label: tk.Label | None = None
        self.mask_threshold_value_label: tk.Label | None = None
        self._mask_threshold_default_bg = ""
        self._mask_threshold_default_fg = ""
        self._startup_source_input = (
            initial_source_input.expanduser().resolve()
            if isinstance(initial_source_input, Path)
            else None
        )
        self._startup_params: dict[str, Any] = dict(initial_params or {})
        self._startup_preload_applied = False

        self._build_ui()
        self._set_default_paths()
        self.root.after(30, self._auto_prepare_model_after_startup)
        self.root.after(80, self._apply_startup_preload)

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=8)

        row1 = tk.Frame(top)
        row1.pack(fill=tk.X)
        tk.Label(row1, text="Encoder ONNX:").pack(side=tk.LEFT)
        tk.Entry(row1, textvariable=self.encoder_var, width=90).pack(side=tk.LEFT, padx=6)
        tk.Button(row1, text="Browse", command=self._browse_encoder).pack(side=tk.LEFT)

        row2 = tk.Frame(top)
        row2.pack(fill=tk.X, pady=4)
        tk.Label(row2, text="Decoder ONNX:").pack(side=tk.LEFT)
        tk.Entry(row2, textvariable=self.decoder_var, width=90).pack(side=tk.LEFT, padx=6)
        tk.Button(row2, text="Browse", command=self._browse_decoder).pack(side=tk.LEFT)

        row3 = tk.Frame(top)
        row3.pack(fill=tk.X)
        tk.Label(row3, text="Model:").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(
            row3,
            textvariable=self.model_name_var,
            values=available_model_names(),
            state="readonly",
            width=36,
        )
        self.model_combo.pack(side=tk.LEFT, padx=6)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)

        tk.Button(row3, text="Load Image", command=self._load_image).pack(side=tk.LEFT, padx=6)
        self.prev_image_btn = tk.Button(
            row3,
            text="Prev",
            command=self._navigate_prev_image,
            state=tk.DISABLED,
        )
        self.prev_image_btn.pack(side=tk.LEFT, padx=3)
        self.next_image_btn = tk.Button(
            row3,
            text="Next",
            command=self._navigate_next_image,
            state=tk.DISABLED,
        )
        self.next_image_btn.pack(side=tk.LEFT, padx=3)
        self.cache_all_btn = tk.Button(
            row3,
            text="Cache Folder (All)",
            command=self._cache_folder_all,
        )
        self.cache_all_btn.pack(side=tk.LEFT, padx=3)
        tk.Button(row3, text="Clear Prompts", command=self._clear_prompts).pack(side=tk.LEFT, padx=6)

        row4 = tk.Frame(top)
        row4.pack(fill=tk.X, pady=4)
        self.mask_threshold_label = tk.Label(row4, text="Mask threshold:")
        self.mask_threshold_label.pack(side=tk.LEFT)
        tk.Scale(
            row4,
            from_=-20.0,
            to=20.0,
            orient=tk.HORIZONTAL,
            resolution=0.01,
            length=260,
            variable=self.mask_threshold_var,
            command=self._on_threshold_changed,
        ).pack(side=tk.LEFT, padx=6)
        self.mask_threshold_value_label = tk.Label(
            row4,
            textvariable=self.mask_threshold_text_var,
            width=6,
            anchor="w",
        )
        self.mask_threshold_value_label.pack(side=tk.LEFT)
        self._mask_threshold_default_bg = str(self.mask_threshold_value_label.cget("bg"))
        self._mask_threshold_default_fg = str(self.mask_threshold_value_label.cget("fg"))
        self._sync_threshold_display()
        tk.Button(row4, text="Reset threshold", command=self._reset_mask_threshold).pack(side=tk.LEFT, padx=(8, 0))
        tk.Checkbutton(
            row4,
            text="Continuous mask only",
            variable=self.continuous_mask_only_var,
            command=self._on_continuous_mask_only_changed,
        ).pack(side=tk.LEFT, padx=(12, 0))
        tk.Checkbutton(
            row4,
            text="Add outline",
            variable=self.selection_outline_var,
            command=self._on_selection_outline_changed,
        ).pack(side=tk.LEFT, padx=(12, 0))
        tk.Label(row4, text="Outline:").pack(side=tk.LEFT, padx=(8, 2))
        self.outline_color_combo = ttk.Combobox(
            row4,
            textvariable=self.selection_outline_color_var,
            values=[name for name, _color in OUTLINE_COLOR_CHOICES],
            state="readonly",
            width=12,
        )
        self.outline_color_combo.pack(side=tk.LEFT)
        self.outline_color_combo.bind("<<ComboboxSelected>>", self._on_selection_outline_changed)

        self.canvas = tk.Canvas(self.root, bg="#222")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        status = tk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Leave>", self._on_mouse_leave)
        self.canvas.bind("<Button-1>", self._on_left_click)   # positive prompt
        self.canvas.bind("<Button-3>", self._on_right_click)  # erase nearby point(s)
        self.canvas.bind("<ButtonPress-2>", self._on_middle_mouse_down)
        self.canvas.bind("<B2-Motion>", self._on_middle_mouse_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_middle_mouse_up)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        self.root.bind("n", lambda _e: self._start_new_mask_session())
        self.root.bind("c", lambda _e: self._clear_prompts())
        self.root.bind("<Left>", lambda _e: self._navigate_prev_image())
        self.root.bind("<Right>", lambda _e: self._navigate_next_image())
        self.root.bind("<space>", lambda _e: self._navigate_next_image())
        self.root.bind("a", lambda _e: self._navigate_prev_image())
        self.root.bind("A", lambda _e: self._navigate_prev_image())
        self.root.bind("d", lambda _e: self._navigate_next_image())
        self.root.bind("D", lambda _e: self._navigate_next_image())

    def _set_default_paths(self) -> None:
        encoder_path, decoder_path = model_weight_paths(
            self.model_name_var.get().strip() or DEFAULT_MODEL_NAME,
            MODELS_DIR,
        )
        self.encoder_var.set(str(encoder_path))
        self.decoder_var.set(str(decoder_path))

    def _on_model_selected(self, _event: tk.Event) -> None:
        self._set_default_paths()
        self.status_var.set(f"Selected model: {self.model_name_var.get()}. Preparing...")
        self._ensure_selected_model_ready_and_loaded()

    def _auto_prepare_model_after_startup(self) -> None:
        self.status_var.set("Preparing default SAM model...")
        self._ensure_selected_model_ready_and_loaded()

    def _apply_startup_preload(self) -> None:
        if self._startup_preload_applied:
            return
        self._startup_preload_applied = True

        try:
            self._apply_gui_param_overrides(self._startup_params)

            startup_source = self._startup_source_input
            if startup_source is None:
                startup_source = resolve_preload_source_input(self._startup_params)

            if startup_source is not None:
                if not startup_source.exists():
                    raise RuntimeError(f"Startup source input does not exist: {startup_source}")
                self._load_image_from_path(startup_source)
        except Exception as exc:
            messagebox.showwarning("Startup preload warning", str(exc))
            self.status_var.set(f"Startup preload warning: {exc}")

    def _apply_gui_param_overrides(self, params: dict[str, Any]) -> None:
        if not params:
            return

        model_name = str(params.get("model_name", "")).strip()
        if model_name:
            if model_name in available_model_names():
                self.model_name_var.set(model_name)
                self._set_default_paths()
            else:
                self.status_var.set(f"Unknown model_name '{model_name}', using default.")

        encoder_path = str(params.get("encoder_path", "")).strip()
        if encoder_path:
            self.encoder_var.set(str(Path(encoder_path).expanduser().resolve()))

        decoder_path = str(params.get("decoder_path", "")).strip()
        if decoder_path:
            self.decoder_var.set(str(Path(decoder_path).expanduser().resolve()))

        threshold_raw = params.get("mask_threshold", params.get("threshold"))
        threshold = parse_float_like(threshold_raw)
        if threshold is not None:
            threshold_clipped = float(np.clip(threshold, -20.0, 20.0))
            self.mask_threshold_var.set(threshold_clipped)
            self._sync_threshold_display()

        for key in ("continuous_mask_only", "connected_component_only"):
            if key not in params:
                continue
            continuous = parse_bool_like(params.get(key))
            if continuous is not None:
                self.continuous_mask_only_var.set(bool(continuous))
            break

        for key in ("add_outline", "selection_outline", "mask_outline"):
            if key not in params:
                continue
            add_outline = parse_bool_like(params.get(key))
            if add_outline is not None:
                self.selection_outline_var.set(bool(add_outline))
            break

        outline_color = str(
            params.get("outline_color", params.get("selection_outline_color", ""))
        ).strip()
        if outline_color:
            normalized = self._normalize_outline_color_name(outline_color)
            if normalized is not None:
                self.selection_outline_color_var.set(normalized)
            else:
                self.status_var.set(f"Unknown outline_color '{outline_color}', using default.")

    def _ensure_selected_model_ready_and_loaded(self) -> bool:
        model_name = self.model_name_var.get().strip() or DEFAULT_MODEL_NAME
        encoder = Path(self.encoder_var.get().strip())
        decoder = Path(self.decoder_var.get().strip())

        if not encoder.exists() or not decoder.exists():
            try:
                self.status_var.set(f"Model files missing. Downloading {model_name}...")
                self.root.update_idletasks()
                encoder, decoder = download_selected_model(
                    model_name=model_name,
                    models_dir=MODELS_DIR,
                )
                self.encoder_var.set(str(encoder))
                self.decoder_var.set(str(decoder))
            except Exception as exc:
                messagebox.showwarning(
                    "Model download failed",
                    f"Could not download selected model '{model_name}':\n{exc}",
                )
                self.status_var.set("Model download failed.")
                return False

        return self._load_model()

    def _on_threshold_changed(self, _value: str) -> None:
        self._sync_threshold_display()
        self._invalidate_committed_overlay_cache()
        self._render()

    def _reset_mask_threshold(self) -> None:
        self.mask_threshold_var.set(0.0)
        self._on_threshold_changed("0.00")

    def _sync_threshold_display(self) -> None:
        value = float(self.mask_threshold_var.get())
        self.mask_threshold_text_var.set(f"{value:.2f}")
        is_nonzero = abs(value) > 1e-9
        bg = "#ffff00" if is_nonzero else self._mask_threshold_default_bg
        fg = "#000000" if is_nonzero else self._mask_threshold_default_fg
        for label in (self.mask_threshold_label, self.mask_threshold_value_label):
            if label is not None:
                label.configure(bg=bg, fg=fg)

    def _on_continuous_mask_only_changed(self) -> None:
        self._seed_mask_cache.clear()
        if self.pos_points:
            self._schedule_preview()
        else:
            self._render()

    def _on_selection_outline_changed(self, _event: tk.Event | None = None) -> None:
        self._render()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        steps = self._mouse_wheel_steps(event)
        if steps == 0:
            return
        state = int(getattr(event, "state", 0) or 0)
        shift_down = bool(state & 0x0001)
        ctrl_down = bool(state & 0x0004)
        if ctrl_down:
            self._adjust_zoom(steps, focus_canvas=(float(event.x), float(event.y)))
            return

        if shift_down:
            scroll_px = self._mouse_wheel_scroll_px()
            self._scroll_view(dx=-steps * scroll_px, dy=0.0)
        else:
            self._adjust_threshold(steps)

    @staticmethod
    def _mouse_wheel_steps(event: tk.Event) -> int:
        delta = int(getattr(event, "delta", 0) or 0)
        if delta != 0:
            return int(delta / 120) if abs(delta) >= 120 else (1 if delta > 0 else -1)
        button = int(getattr(event, "num", 0) or 0)
        if button == 4:
            return 1
        if button == 5:
            return -1
        return 0

    def _adjust_threshold(self, steps: int) -> None:
        current = float(self.mask_threshold_var.get())
        updated = float(np.clip(current + 0.25 * steps, -20.0, 20.0))
        self.mask_threshold_var.set(updated)
        self._on_threshold_changed(f"{updated:.2f}")

    def _mouse_wheel_scroll_px(self) -> float:
        canvas_h = max(float(self.canvas.winfo_height()), 1.0)
        return float(np.clip(canvas_h * 0.12, 40.0, 160.0))

    def _scroll_view(self, dx: float, dy: float) -> None:
        if self.image_pil is None:
            return
        old_x = float(self.view_x)
        old_y = float(self.view_y)
        self.view_x = old_x + float(dx)
        self.view_y = old_y + float(dy)
        self._clamp_view()
        if abs(self.view_x - old_x) < 1e-6 and abs(self.view_y - old_y) < 1e-6:
            return
        self._render()

    def _on_middle_mouse_down(self, event: tk.Event) -> str:
        if self.image_pil is None:
            self._middle_pan_last_canvas_xy = None
            return "break"
        self._middle_pan_last_canvas_xy = (float(event.x), float(event.y))
        if self.hover_point is not None or self._hover_preview_mask is not None:
            self.hover_point = None
            self._hover_preview_mask = None
            self._render()
        return "break"

    def _on_middle_mouse_drag(self, event: tk.Event) -> str:
        previous = self._middle_pan_last_canvas_xy
        current = (float(event.x), float(event.y))
        if previous is None:
            self._middle_pan_last_canvas_xy = current
            return "break"
        dx = previous[0] - current[0]
        dy = previous[1] - current[1]
        self._middle_pan_last_canvas_xy = current
        self._scroll_view(dx=dx, dy=dy)
        return "break"

    def _on_middle_mouse_up(self, _event: tk.Event) -> str:
        self._middle_pan_last_canvas_xy = None
        return "break"

    def _adjust_zoom(self, steps: int, focus_canvas: tuple[float, float] | None = None) -> None:
        if self.image_pil is None:
            return
        old_dw, old_dh = self.display_size
        old_view_x = float(self.view_x)
        old_view_y = float(self.view_y)

        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        if focus_canvas is None:
            fx = canvas_w / 2.0
            fy = canvas_h / 2.0
        else:
            fx = float(np.clip(focus_canvas[0], 0.0, max(0.0, canvas_w - 1)))
            fy = float(np.clip(focus_canvas[1], 0.0, max(0.0, canvas_h - 1)))

        iw, ih = self.image_pil.size
        if old_dw <= 1 or old_dh <= 1 or iw <= 0 or ih <= 0:
            fx_img = 0.5
            fy_img = 0.5
        else:
            disp_x = np.clip(old_view_x + fx, 0.0, max(0.0, old_dw - 1))
            disp_y = np.clip(old_view_y + fy, 0.0, max(0.0, old_dh - 1))
            fx_img = float(disp_x / max(1.0, old_dw))
            fy_img = float(disp_y / max(1.0, old_dh))

        updated = float(np.clip(self.zoom_factor * (1.12 ** steps), self._zoom_min, self._zoom_max))
        if abs(updated - self.zoom_factor) < 1e-6:
            return
        self.zoom_factor = updated
        self._prepare_display_base(reset_view=False)

        new_dw, new_dh = self.display_size
        new_focus_x = fx_img * max(1.0, new_dw)
        new_focus_y = fy_img * max(1.0, new_dh)
        self.view_x = new_focus_x - fx
        self.view_y = new_focus_y - fy
        self._clamp_view()

        self._render()
        self.status_var.set(f"Zoom: {self.zoom_factor:.2f}x")

    def _browse_encoder(self) -> None:
        path = filedialog.askopenfilename(title="Select encoder ONNX", filetypes=[("ONNX", "*.onnx"), ("All files", "*.*")])
        if path:
            self.encoder_var.set(path)

    def _browse_decoder(self) -> None:
        path = filedialog.askopenfilename(title="Select decoder ONNX", filetypes=[("ONNX", "*.onnx"), ("All files", "*.*")])
        if path:
            self.decoder_var.set(path)

    def _load_model(self) -> bool:
        encoder = Path(self.encoder_var.get().strip())
        decoder = Path(self.decoder_var.get().strip())
        if not encoder.exists() or not decoder.exists():
            self.status_var.set("Model load failed: missing files.")
            return False

        model_signature = (str(encoder), str(decoder))
        if self.model is not None and self._loaded_model_signature == model_signature:
            self.status_var.set(f"SAM model ready: {self.model_name_var.get()}")
            return True

        try:
            self.status_var.set("Loading SAM ONNX sessions...")
            self.root.update_idletasks()
            self.model = SegmentAnythingOnnx(str(encoder), str(decoder))
            self._loaded_model_signature = model_signature
            self.status_var.set("SAM model loaded.")
            if self.image_np is not None:
                self._ensure_embedding_for_current_image(show_status=False)
            return True
        except Exception as exc:
            self.model = None
            self._loaded_model_signature = None
            messagebox.showerror("Model load failed", str(exc))
            self.status_var.set("Model load failed.")
            return False

    def _load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        self._load_image_from_path(Path(path))

    def _load_image_from_path(self, image_path: Path) -> bool:
        if self._predict_busy or (self._dirty_preview and bool(self.pos_points)):
            self.status_var.set("Finishing current mask before loading another image.")
            return False
        self._cancel_preview_for_image_transition()

        try:
            image_path = image_path.resolve()
            img = Image.open(image_path).convert("RGB")
            self.image_path = image_path
            self._refresh_folder_image_list(image_path.parent, current_image_path=image_path)
            self.image_pil = img
            self.image_np = np.asarray(img)
            self.zoom_factor = 1.0
            self._prepare_display_base(reset_view=True)
            self._clear_prompts(reset_mask=True, persist=False)
            restored_count = self._load_annotations_for_current_image()
            if self.model is None:
                if not self._ensure_selected_model_ready_and_loaded():
                    if restored_count > 0:
                        self.status_var.set(
                            f"Image loaded with {restored_count} saved mask(s), but SAM model is unavailable."
                        )
                    else:
                        self.status_var.set("Image loaded, but SAM model is unavailable.")
                    self._render()
                    return
            self._ensure_embedding_for_current_image(show_status=True)
            if restored_count > 0:
                self.status_var.set(
                    f"Image loaded. Restored {restored_count} saved mask(s). Move mouse / click prompts."
                )
            else:
                self.status_var.set("Image loaded. Move mouse / click prompts.")
            self._render()
            return True
        except Exception as exc:
            messagebox.showerror("Image load failed", str(exc))
            return False

    @staticmethod
    def _is_supported_image_path(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS

    def _refresh_folder_image_list(
        self, folder: Path, current_image_path: Path | None = None
    ) -> None:
        if not folder.exists():
            self._folder_image_paths = []
            self._current_image_index = -1
            self._update_navigation_button_states()
            return

        paths = sorted(
            [p.resolve() for p in folder.iterdir() if self._is_supported_image_path(p)],
            key=lambda p: p.name.lower(),
        )
        self._folder_image_paths = paths
        self._current_image_index = -1

        if current_image_path is not None:
            target = str(current_image_path.resolve()).lower()
            for idx, path in enumerate(paths):
                if str(path).lower() == target:
                    self._current_image_index = idx
                    break

        if self._current_image_index < 0 and paths:
            self._current_image_index = 0
        self._update_navigation_button_states()

    def _update_navigation_button_states(self) -> None:
        if self.prev_image_btn is not None:
            enable_prev = (
                (not self._batch_cache_busy)
                and self._current_image_index > 0
                and len(self._folder_image_paths) > 1
            )
            self.prev_image_btn.configure(state=(tk.NORMAL if enable_prev else tk.DISABLED))
        if self.next_image_btn is not None:
            enable_next = (
                (not self._batch_cache_busy)
                and 0 <= self._current_image_index < (len(self._folder_image_paths) - 1)
                and len(self._folder_image_paths) > 1
            )
            self.next_image_btn.configure(state=(tk.NORMAL if enable_next else tk.DISABLED))
        if self.cache_all_btn is not None:
            self.cache_all_btn.configure(
                state=(tk.DISABLED if self._batch_cache_busy else tk.NORMAL)
            )

    def _cancel_preview_for_image_transition(self) -> None:
        self._dirty_preview = False
        self._request_id += 1
        self._seed_mask_cache.clear()
        self.hover_point = None
        self._hover_preview_mask = None
        self.last_mask = None

    def _defer_navigation_if_prediction_active(self, delta: int) -> bool:
        if self._predict_busy or (self._dirty_preview and bool(self.pos_points)):
            self._pending_navigation_delta = int(delta)
            self.status_var.set("Finishing current mask before changing images...")
            if self._dirty_preview and not self._predict_busy:
                self._run_preview()
            return True
        if self._dirty_preview:
            self._cancel_preview_for_image_transition()
        return False

    def _consume_pending_navigation(self) -> None:
        delta = self._pending_navigation_delta
        self._pending_navigation_delta = None
        if delta is None:
            return
        self._cancel_preview_for_image_transition()
        self._navigate_image_delta(int(delta))

    def _navigate_prev_image(self) -> None:
        self._navigate_image_delta(-1)

    def _navigate_next_image(self) -> None:
        self._navigate_image_delta(1)

    def _navigate_image_delta(self, delta: int) -> None:
        if delta == 0:
            return
        if self._defer_navigation_if_prediction_active(delta):
            return
        if self.image_path is not None and not self._folder_image_paths:
            self._refresh_folder_image_list(self.image_path.parent, self.image_path)
        if not self._folder_image_paths:
            self.status_var.set("No image folder list available. Load an image first.")
            return
        if self.image_path is None:
            self._load_image_from_path(self._folder_image_paths[0])
            return

        if delta < 0:
            if self._current_image_index <= 0:
                self.status_var.set("Already at first image in folder.")
                return
            next_path = self._folder_image_paths[self._current_image_index - 1]
        else:
            if (
                self._current_image_index < 0
                or self._current_image_index >= len(self._folder_image_paths) - 1
            ):
                self.status_var.set("Already at last image in folder.")
                return
            next_path = self._folder_image_paths[self._current_image_index + 1]
        self._load_image_from_path(next_path)

    def _cache_folder_all(self) -> None:
        self._cache_folder_embeddings()

    def _cache_folder_embeddings(self) -> None:
        if self._batch_cache_busy:
            self.status_var.set("Folder caching is already running.")
            return

        folder: Path | None = None
        if self.image_path is not None:
            folder = self.image_path.parent
        else:
            selected = filedialog.askdirectory(title="Select image folder to cache")
            if selected:
                folder = Path(selected).resolve()
        if folder is None:
            return

        self._refresh_folder_image_list(folder, current_image_path=self.image_path)
        candidates = list(self._folder_image_paths)
        if not candidates:
            self.status_var.set("No supported images found in folder.")
            return

        if not self._ensure_selected_model_ready_and_loaded():
            self.status_var.set("Cannot cache embeddings: SAM model is unavailable.")
            return

        encoder = Path(self.encoder_var.get().strip())
        decoder = Path(self.decoder_var.get().strip())
        model_cache_token = self._model_cache_token()
        model_cache_metadata = self._model_cache_metadata()
        model_image_size = int(self.model.image_size) if self.model is not None else 1024
        total = len(candidates)

        self._batch_cache_busy = True
        self._update_navigation_button_states()
        self.status_var.set(f"Caching embeddings for {total} image(s)...")

        def worker() -> None:
            cached = 0
            skipped = 0
            failed = 0
            try:
                batch_model = SegmentAnythingOnnx(
                    str(encoder), str(decoder), image_size=model_image_size
                )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda: self._finish_folder_caching(
                        total=total,
                        cached=0,
                        skipped=0,
                        failed=total,
                        note=f"Could not initialize model for batch cache: {exc}",
                    ),
                )
                return

            for idx, path in enumerate(candidates, start=1):
                try:
                    if self._is_embedding_cache_fresh_for_image(
                        path,
                        model_cache_token,
                        model_cache_metadata=model_cache_metadata,
                    ):
                        skipped += 1
                    else:
                        image_np = np.asarray(Image.open(path).convert("RGB"))
                        batch_model.set_image(image_np)
                        embedding = batch_model.get_embedding()
                        if embedding is None:
                            raise RuntimeError("No embedding returned.")
                        if self._save_embedding_cache_for_image(
                            image_path=path,
                            image_np=image_np,
                            embedding=embedding,
                            model_cache_token=model_cache_token,
                            model_metadata=model_cache_metadata,
                        ):
                            cached += 1
                        else:
                            failed += 1
                except Exception:
                    failed += 1

                self.root.after(
                    0,
                    lambda i=idx, t=total, c=cached, s=skipped, f=failed: self.status_var.set(
                        f"Caching embeddings {i}/{t} (new {c}, skipped {s}, failed {f})"
                    ),
                )

            self.root.after(
                0,
                lambda: self._finish_folder_caching(
                    total=total,
                    cached=cached,
                    skipped=skipped,
                    failed=failed,
                    note=None,
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _finish_folder_caching(
        self,
        total: int,
        cached: int,
        skipped: int,
        failed: int,
        note: str | None = None,
    ) -> None:
        self._batch_cache_busy = False
        self._update_navigation_button_states()
        if note:
            self.status_var.set(note)
            return
        self.status_var.set(
            f"Folder caching complete: {cached} new, {skipped} already cached, {failed} failed (total {total})."
        )

    def _annotation_store_path_for_image(self, image_path: Path) -> Path:
        return image_path.with_name(f"{image_path.name}{ANNOTATION_FILE_SUFFIX}")

    def _annotation_metadata_path_for_image(self, image_path: Path) -> Path:
        return image_path.with_name(f"{image_path.name}{ANNOTATION_METADATA_FILE_SUFFIX}")

    def _annotation_store_path(self) -> Path | None:
        if self.image_path is None:
            return None
        return self._annotation_store_path_for_image(self.image_path)

    def _annotation_metadata_path(self) -> Path | None:
        if self.image_path is None:
            return None
        return self._annotation_metadata_path_for_image(self.image_path)

    def _model_cache_token(self) -> str:
        model_name = self.model_name_var.get().strip() or DEFAULT_MODEL_NAME
        encoder, decoder = self._loaded_model_signature or (
            self.encoder_var.get().strip(),
            self.decoder_var.get().strip(),
        )
        image_size = int(self.model.image_size) if self.model is not None else 0
        return build_model_cache_token(
            model_name=model_name,
            encoder_path=Path(encoder),
            decoder_path=Path(decoder),
            image_size=image_size,
        )

    def _model_cache_metadata(self) -> dict[str, Any]:
        model_name = self.model_name_var.get().strip() or DEFAULT_MODEL_NAME
        encoder, decoder = self._loaded_model_signature or (
            self.encoder_var.get().strip(),
            self.decoder_var.get().strip(),
        )
        image_size = int(self.model.image_size) if self.model is not None else 0
        return build_model_cache_metadata(
            model_name=model_name,
            encoder_path=Path(encoder),
            decoder_path=Path(decoder),
            image_size=image_size,
        )

    def _embedding_store_path_for_image(
        self, image_path: Path, model_cache_token: str | None = None
    ) -> Path:
        token = model_cache_token if model_cache_token is not None else self._model_cache_token()
        token_hash = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
        return image_path.with_name(
            f"{image_path.name}.{token_hash}{EMBEDDING_FILE_SUFFIX}"
        )

    def _embedding_store_path(self) -> Path | None:
        if self.image_path is None:
            return None
        return self._embedding_store_path_for_image(self.image_path)

    def _ensure_embedding_for_current_image(self, show_status: bool = True) -> bool:
        if self.model is None or self.image_np is None:
            return False
        if self._load_cached_embedding_for_current_image():
            if show_status:
                self.status_var.set("Loaded cached image embedding.")
            return True
        if show_status:
            self.status_var.set("Computing image embedding...")
        self.root.update_idletasks()
        self.model.set_image(self.image_np)
        self._save_cached_embedding_for_current_image()
        return True

    def _save_embedding_cache_for_image(
        self,
        image_path: Path,
        image_np: np.ndarray,
        embedding: np.ndarray,
        model_cache_token: str,
        model_metadata: dict[str, Any] | None = None,
    ) -> bool:
        cache_path = self._embedding_store_path_for_image(
            image_path=image_path,
            model_cache_token=model_cache_token,
        )
        h, w = image_np.shape[:2]
        image_file_size = None
        image_mtime_ns = None
        try:
            stat = image_path.stat()
            image_file_size = int(stat.st_size)
            image_mtime_ns = int(stat.st_mtime_ns)
        except Exception:
            image_file_size = None
            image_mtime_ns = None

        payload: dict[str, Any] = {
            "version": 1,
            "saved_at_utc": self._utc_now_iso(),
            "model_cache_token": model_cache_token,
            "image_path": str(image_path),
            "image_name": image_path.name,
            "image_size_hw": [int(h), int(w)],
            "image_file_size_bytes": image_file_size,
            "image_mtime_ns": image_mtime_ns,
        }
        payload.update(dict(model_metadata or {}))

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with tmp_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    embedding=np.asarray(embedding, dtype=np.float32),
                    metadata_json=np.asarray(json.dumps(payload), dtype=np.str_),
                )
            tmp_path.replace(cache_path)
            return True
        except Exception:
            return False

    def _is_embedding_cache_fresh_for_image(
        self,
        image_path: Path,
        model_cache_token: str,
        model_cache_metadata: dict[str, Any] | None = None,
    ) -> bool:
        candidates = [path for path in embedding_cache_candidate_paths_for_image(
            image_path,
            model_cache_token,
        ) if path.exists()]
        model_metadata = dict(model_cache_metadata or self._model_cache_metadata())
        allow_unknown_legacy = len(candidates) == 1
        for cache_path in candidates:
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    metadata_json = (
                        data["metadata_json"] if "metadata_json" in data.files else None
                    )
            except Exception:
                continue
            if metadata_json is None:
                continue

            try:
                if np.ndim(metadata_json) == 0:
                    metadata_text = str(metadata_json.item())
                else:
                    metadata_text = str(np.asarray(metadata_json).reshape(-1)[0])
                payload = json.loads(metadata_text)
                if not isinstance(payload, dict):
                    continue
            except Exception:
                continue

            if not embedding_cache_payload_matches_image(payload, image_path):
                continue
            if not embedding_cache_payload_matches_model(
                payload,
                model_cache_token,
                model_metadata,
                allow_legacy_unknown_token=allow_unknown_legacy,
            ):
                continue
            return True
        return False
    def _save_cached_embedding_for_current_image(self) -> bool:
        if self.model is None or self.image_np is None or self.image_path is None:
            return False
        embedding = self.model.get_embedding()
        if embedding is None:
            return False
        return self._save_embedding_cache_for_image(
            image_path=self.image_path,
            image_np=self.image_np,
            embedding=embedding,
            model_cache_token=self._model_cache_token(),
            model_metadata=self._model_cache_metadata(),
        )

    def _load_cached_embedding_for_current_image(self) -> bool:
        if self.model is None or self.image_np is None or self.image_path is None:
            return False

        model_cache_token = self._model_cache_token()
        model_metadata = self._model_cache_metadata()
        expected_cache_path = self._embedding_store_path_for_image(
            self.image_path,
            model_cache_token=model_cache_token,
        )
        candidates = [path for path in embedding_cache_candidate_paths_for_image(
            self.image_path,
            model_cache_token,
        ) if path.exists()]
        if not candidates:
            return False

        allow_unknown_legacy = len(candidates) == 1
        for cache_path in candidates:
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    embedding = np.asarray(data["embedding"], dtype=np.float32)
                    metadata_json = (
                        data["metadata_json"] if "metadata_json" in data.files else None
                    )
            except Exception:
                continue

            if embedding.ndim != 4:
                continue

            payload: dict[str, Any] = {}
            if metadata_json is not None:
                try:
                    if np.ndim(metadata_json) == 0:
                        metadata_text = str(metadata_json.item())
                    else:
                        metadata_text = str(np.asarray(metadata_json).reshape(-1)[0])
                    if metadata_text:
                        loaded_payload = json.loads(metadata_text)
                        if isinstance(loaded_payload, dict):
                            payload = loaded_payload
                except Exception:
                    payload = {}

            if not payload:
                if cache_path != expected_cache_path:
                    continue
            else:
                if not embedding_cache_payload_matches_image(
                    payload,
                    self.image_path,
                    image_shape_hw=self.image_np.shape[:2],
                ):
                    continue
                if not embedding_cache_payload_matches_model(
                    payload,
                    model_cache_token,
                    model_metadata,
                    allow_legacy_unknown_token=allow_unknown_legacy,
                ):
                    continue

            self.model.set_image_with_embedding(self.image_np, embedding)
            if cache_path != expected_cache_path:
                self._save_cached_embedding_for_current_image()
            return True
        return False
    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def _record_click_metadata(self, session_id: int, seed_point: tuple[float, float]) -> None:
        if self.image_np is None:
            return
        now = self._utc_now_iso()
        h, w = self.image_np.shape[:2]
        self._session_metadata[int(session_id)] = {
            "session_id": int(session_id),
            "seed_point_xy": [float(seed_point[0]), float(seed_point[1])],
            "seed_point_xy_round": [int(round(seed_point[0])), int(round(seed_point[1]))],
            "created_at_utc": now,
            "last_updated_utc": now,
            "image_size_hw": [int(h), int(w)],
            "model_name": self.model_name_var.get().strip() or DEFAULT_MODEL_NAME,
            "threshold_at_click": float(self.mask_threshold_var.get()),
            "continuous_mask_only_at_click": bool(self.continuous_mask_only_var.get()),
            "zoom_factor_at_click": float(self.zoom_factor),
            "view_origin_xy": [float(self.view_x), float(self.view_y)],
        }

    def _update_session_metadata_with_mask(self, session_id: int, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=np.float32)
        threshold = float(self.mask_threshold_var.get())
        binary = np.asarray(mask > threshold)
        area_px = int(np.count_nonzero(binary))
        bbox_xyxy: list[int] | None = None
        if area_px > 0:
            ys, xs = np.nonzero(binary)
            bbox_xyxy = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            ]
        md = dict(self._session_metadata.get(int(session_id), {}))
        md.update(
            {
                "session_id": int(session_id),
                "last_updated_utc": self._utc_now_iso(),
                "threshold_last_saved": threshold,
                "continuous_mask_only_last_saved": bool(self.continuous_mask_only_var.get()),
                "mask_values_are_binary": False,
                "mask_values_kind": "sam_logits",
                "mask_area_px": area_px,
                "mask_bbox_xyxy": bbox_xyxy,
                "mask_logit_min": float(np.min(mask)),
                "mask_logit_max": float(np.max(mask)),
                "mask_logit_mean": float(np.mean(mask)),
            }
        )
        self._session_metadata[int(session_id)] = md

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        mask_engine.write_json_atomic(path, payload)

    @staticmethod
    def _load_json_object_optional(path: Path | None) -> dict[str, Any]:
        return mask_engine.load_json_object_optional(path)

    def _sessions_by_mask_index(self) -> dict[int, int]:
        return mask_engine.sessions_by_mask_index(
            self._session_to_mask_index,
            len(self.committed_masks),
        )

    def _session_metadata_for_mask_index(self, idx: int) -> dict[str, Any]:
        return mask_engine.session_metadata_for_mask_index(
            idx,
            self._session_to_mask_index,
            self._session_metadata,
            len(self.committed_masks),
        )

    @staticmethod
    def _metadata_marks_binary_mask(md: dict[str, Any]) -> bool:
        return mask_engine.metadata_marks_binary_mask(md)

    @staticmethod
    def _array_values_are_binary(mask: np.ndarray) -> bool:
        return mask_engine.array_values_are_binary(mask)

    @staticmethod
    def _threshold_from_metadata(md: dict[str, Any], fallback: float) -> float:
        return mask_engine.threshold_from_metadata(md, fallback)

    def _threshold_for_committed_mask(self, idx: int, fallback: float) -> float:
        return mask_engine.threshold_for_committed_mask(
            idx,
            fallback,
            self._session_to_mask_index,
            self._session_metadata,
            len(self.committed_masks),
        )

    def _threshold_for_ignore_mask(self, idx: int, fallback: float) -> float:
        return mask_engine.threshold_for_ignore_mask(
            idx,
            fallback,
            getattr(self, "_ignore_mask_metadata", []),
        )

    def _thresholds_for_committed_masks(self, fallback: float) -> list[float]:
        return mask_engine.thresholds_for_committed_masks(
            self.committed_masks,
            fallback,
            self._session_to_mask_index,
            self._session_metadata,
        )

    def _thresholds_for_ignore_masks(self, fallback: float) -> list[float]:
        return mask_engine.thresholds_for_ignore_masks(
            list(getattr(self, "ignore_masks", [])),
            fallback,
            list(getattr(self, "_ignore_mask_metadata", [])),
        )

    def _binary_ignore_mask_at_index(self, idx: int, fallback: float) -> np.ndarray:
        return mask_engine.binary_ignore_mask_at_index(
            list(getattr(self, "ignore_masks", [])),
            list(getattr(self, "_ignore_mask_metadata", [])),
            idx,
            fallback,
        )

    def _combined_ignore_mask(self, fallback: float) -> np.ndarray | None:
        return mask_engine.combined_ignore_mask(
            list(getattr(self, "ignore_masks", [])),
            list(getattr(self, "_ignore_mask_metadata", [])),
            fallback,
        )

    def _binary_mask_with_ignore(
        self,
        mask: np.ndarray,
        threshold: float,
        subtract_ignored: bool = True,
    ) -> np.ndarray:
        return mask_engine.binary_mask_with_ignore(
            mask,
            threshold,
            ignore_masks=list(getattr(self, "ignore_masks", [])),
            ignore_mask_metadata=list(getattr(self, "_ignore_mask_metadata", [])),
            subtract_ignored=subtract_ignored,
        )

    def _binary_committed_mask_at_index(self, idx: int, fallback: float) -> np.ndarray:
        return mask_engine.binary_committed_mask_at_index(
            self.committed_masks,
            idx,
            fallback,
            self._session_to_mask_index,
            self._session_metadata,
            ignore_masks=list(getattr(self, "ignore_masks", [])),
            ignore_mask_metadata=list(getattr(self, "_ignore_mask_metadata", [])),
        )

    @staticmethod
    def _mask_area_and_bbox(binary: np.ndarray) -> tuple[int, list[int] | None]:
        return mask_engine.mask_area_and_bbox(binary)

    def _build_annotation_payload(self, saved_at: str) -> dict[str, Any]:
        h, w = self.image_np.shape[:2]
        return mask_engine.build_annotation_payload(
            saved_at=saved_at,
            image_path=self.image_path,
            image_shape_hw=(int(h), int(w)),
            model_name=self.model_name_var.get().strip() or DEFAULT_MODEL_NAME,
            mask_threshold=float(self.mask_threshold_var.get()),
            continuous_mask_only=bool(self.continuous_mask_only_var.get()),
            committed_masks=self.committed_masks,
            ignore_masks=list(getattr(self, "ignore_masks", [])),
            session_to_mask_index=self._session_to_mask_index,
            session_metadata=self._session_metadata,
            ignore_mask_metadata=list(getattr(self, "_ignore_mask_metadata", [])),
        )

    def _save_annotations_for_current_image(self) -> bool:
        save_path = self._annotation_store_path()
        metadata_path = self._annotation_metadata_path()
        if save_path is None or self.image_np is None:
            return False
        try:
            h, w = self.image_np.shape[:2]
            mask_engine.save_annotations(
                save_path=save_path,
                metadata_path=metadata_path,
                image_path=self.image_path,
                image_shape_hw=(int(h), int(w)),
                model_name=self.model_name_var.get().strip() or DEFAULT_MODEL_NAME,
                mask_threshold=float(self.mask_threshold_var.get()),
                continuous_mask_only=bool(self.continuous_mask_only_var.get()),
                committed_masks=self.committed_masks,
                ignore_masks=list(getattr(self, "ignore_masks", [])),
                session_to_mask_index=self._session_to_mask_index,
                session_metadata=self._session_metadata,
                ignore_mask_metadata=list(getattr(self, "_ignore_mask_metadata", [])),
                saved_at=self._utc_now_iso(),
            )
            return True
        except Exception as exc:
            self.status_var.set(f"Warning: autosave failed ({exc})")
            return False

    def _load_annotations_for_current_image(self) -> int:
        save_path = self._annotation_store_path()
        metadata_path = self._annotation_metadata_path()

        def clear_saved_state() -> None:
            self._session_metadata.clear()
            self._ignore_mask_metadata.clear()
            self.ignore_masks.clear()

        if save_path is None or self.image_np is None:
            clear_saved_state()
            return 0
        if not save_path.exists():
            clear_saved_state()
            return 0

        h, w = self.image_np.shape[:2]
        try:
            loaded = mask_engine.load_annotations(
                save_path=save_path,
                metadata_path=metadata_path,
                image_shape_hw=(int(h), int(w)),
            )
        except Exception as exc:
            message = str(exc)
            if message == "unexpected mask shape":
                self.status_var.set("Warning: saved annotations ignored (unexpected mask shape).")
            elif message == "unexpected ignore mask shape":
                self.status_var.set(
                    "Warning: saved annotations ignored (unexpected ignore mask shape)."
                )
            elif message == "mask size mismatch":
                self.status_var.set("Saved annotations found, but size does not match this image.")
            elif message == "ignore mask size mismatch":
                self.status_var.set(
                    "Saved ignore annotations found, but size does not match this image."
                )
            else:
                self.status_var.set(f"Warning: could not load saved annotations ({exc})")
            clear_saved_state()
            return 0

        self.last_mask = None
        self.pos_points.clear()
        self.neg_points.clear()
        self._seed_mask_cache.clear()
        self.hover_point = None
        self._hover_preview_mask = None
        self._auto_advance_session_id = None
        self.committed_masks.clear()
        self.ignore_masks.clear()
        self._ignore_mask_metadata.clear()
        self._session_to_mask_index.clear()
        self._session_metadata.clear()

        self.committed_masks.extend(loaded.masks)
        self.ignore_masks.extend(loaded.ignore_masks)
        self._ignore_mask_metadata.extend(loaded.ignore_mask_metadata)
        self._session_to_mask_index.update(loaded.session_to_mask_index)
        self._session_metadata.update(loaded.session_metadata)
        self._active_session_id = int(loaded.active_session_id)
        self._invalidate_committed_overlay_cache()
        return len(self.committed_masks)
    def _prepare_display_base(self, reset_view: bool = False) -> None:
        if self.image_pil is None:
            return
        canvas_w = max(self.canvas.winfo_width(), 640)
        canvas_h = max(self.canvas.winfo_height(), 480)
        image = self.image_pil
        iw, ih = image.size
        fit_scale = min(canvas_w / iw, canvas_h / ih)
        scale = max(1e-6, fit_scale * self.zoom_factor)
        dw = max(1, int(iw * scale))
        dh = max(1, int(ih * scale))
        resized = image.resize((dw, dh), Image.Resampling.BILINEAR)
        old_display_size = self.display_size
        self.display_base_np = np.asarray(resized)
        self.display_size = (dw, dh)
        if self.display_size != old_display_size:
            self._invalidate_committed_overlay_cache()
        if reset_view:
            self.view_x = 0.0
            self.view_y = 0.0
        self._clamp_view()

    def _clamp_view(self) -> None:
        dw, dh = self.display_size
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        max_x = max(0.0, float(dw - canvas_w))
        max_y = max(0.0, float(dh - canvas_h))
        self.view_x = float(np.clip(self.view_x, 0.0, max_x))
        self.view_y = float(np.clip(self.view_y, 0.0, max_y))

    def _to_image_coords(self, x: float, y: float) -> tuple[float, float] | None:
        if self.image_np is None:
            return None
        iw = self.image_np.shape[1]
        ih = self.image_np.shape[0]
        dw, dh = self.display_size
        if dw <= 1 or dh <= 1:
            return None
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        visible_w = min(dw, canvas_w)
        visible_h = min(dh, canvas_h)
        if x < 0 or y < 0 or x >= visible_w or y >= visible_h:
            return None
        disp_x = float(np.clip(self.view_x + x, 0.0, dw - 1))
        disp_y = float(np.clip(self.view_y + y, 0.0, dh - 1))
        return disp_x * iw / dw, disp_y * ih / dh

    @staticmethod
    def _event_ctrl_down(event: tk.Event) -> bool:
        return bool(int(getattr(event, "state", 0) or 0) & 0x0004)

    def _on_left_click(self, event: tk.Event) -> None:
        p = self._to_image_coords(event.x, event.y)
        if p is None:
            return
        if self._event_ctrl_down(event):
            self._add_ignore_mask_at_point(p)
            return
        # Left click seeds the current object session and auto-advances to
        # the next session once this click's prediction has been accepted.
        if int(self._active_session_id) <= 0:
            self._active_session_id = 1
        session_id = int(self._active_session_id)
        self._auto_advance_session_id = session_id
        self._record_click_metadata(session_id=session_id, seed_point=p)
        self.pos_points[:] = [p]
        self.neg_points.clear()
        self._seed_mask_cache.clear()
        self._hover_preview_mask = None
        self.hover_point = p
        self._schedule_preview()

    def _start_new_mask_session(self) -> None:
        # Enter a fresh object proposal session.
        self._auto_advance_session_id = None
        self._active_session_id += 1
        self.pos_points.clear()
        self.neg_points.clear()
        self._seed_mask_cache.clear()
        self.hover_point = None
        self._hover_preview_mask = None
        self.last_mask = None
        self._render()

    def _on_right_click(self, event: tk.Event) -> None:
        if self.image_np is None:
            return
        if self._event_ctrl_down(event):
            image_pt = self._to_image_coords(float(event.x), float(event.y))
            removed_ignores = 0
            if image_pt is not None:
                removed_ignores = self._remove_ignore_masks_near_image(
                    image_center=image_pt,
                    radius_image_px=self._display_radius_to_image_radius(
                        self._erase_radius_display_px
                    ),
                )
            if removed_ignores > 0:
                self._save_annotations_for_current_image()
                self._render()
                self.status_var.set(f"Removed {removed_ignores} ignored region(s).")
            else:
                self.status_var.set("No ignored region nearby.")
            return
        annotations_changed = False
        removed_points = self._remove_points_near_display(
            self.pos_points,
            canvas_xy=(float(event.x), float(event.y)),
            radius_display_px=self._erase_radius_display_px,
        )
        # Keep backward-compat behavior sane if any negative points still exist.
        removed_points += self._remove_points_near_display(
            self.neg_points,
            canvas_xy=(float(event.x), float(event.y)),
            radius_display_px=self._erase_radius_display_px,
        )

        removed_masks = 0
        image_pt = self._to_image_coords(float(event.x), float(event.y))
        if image_pt is not None:
            removed_masks = self._remove_committed_masks_near_image(
                image_center=image_pt,
                radius_image_px=self._display_radius_to_image_radius(
                    self._erase_radius_display_px
                ),
            )
        if removed_masks > 0:
            annotations_changed = True
            self.last_mask = None
            self._hover_preview_mask = None

        if removed_points > 0:
            self._hover_preview_mask = None
            self._seed_mask_cache.clear()
            session_id = int(self._active_session_id)
            if not self.pos_points and session_id > 0:
                # No seed points remain for this object proposal; drop its mask.
                self.last_mask = None
                had_mask = session_id in self._session_to_mask_index
                self._remove_committed_mask_for_session(session_id)
                annotations_changed = annotations_changed or had_mask
                self._render()
            else:
                self._schedule_preview()
        elif removed_masks > 0:
            self._render()

        if annotations_changed:
            self._save_annotations_for_current_image()

        if removed_points > 0 or removed_masks > 0:
            parts: list[str] = []
            if removed_points > 0:
                parts.append(f"{removed_points} point(s)")
            if removed_masks > 0:
                parts.append(f"{removed_masks} mask(s)")
            self.status_var.set(f"Removed {' and '.join(parts)}.")
        else:
            self.status_var.set("No point or mask near cursor to remove.")

    def _image_point_to_canvas(
        self, point: tuple[float, float]
    ) -> tuple[float, float] | None:
        if self.image_np is None:
            return None
        iw = self.image_np.shape[1]
        ih = self.image_np.shape[0]
        dw, dh = self.display_size
        if dw <= 1 or dh <= 1:
            return None
        x_full = float(point[0]) * float(dw) / max(1.0, float(iw))
        y_full = float(point[1]) * float(dh) / max(1.0, float(ih))
        return float(x_full - self.view_x), float(y_full - self.view_y)

    def _remove_points_near_display(
        self,
        points: list[tuple[float, float]],
        canvas_xy: tuple[float, float],
        radius_display_px: float,
    ) -> int:
        cx, cy = float(canvas_xy[0]), float(canvas_xy[1])
        r2 = float(radius_display_px) ** 2
        kept: list[tuple[float, float]] = []
        removed = 0
        for px, py in points:
            canvas_pt = self._image_point_to_canvas((float(px), float(py)))
            if canvas_pt is None:
                kept.append((float(px), float(py)))
                continue
            if (canvas_pt[0] - cx) ** 2 + (canvas_pt[1] - cy) ** 2 <= r2:
                removed += 1
            else:
                kept.append((float(px), float(py)))
        points[:] = kept
        return removed

    def _add_ignore_mask_at_point(self, point: tuple[float, float]) -> None:
        if self.model is None or self.image_np is None:
            self.status_var.set("Load an image/model before adding ignored regions.")
            return
        with self._predict_lock:
            if self._predict_busy:
                self.status_var.set("Busy predicting; try ignore click again in a moment.")
                return
            self._predict_busy = True

        seed_point = (float(point[0]), float(point[1]))
        threshold_at_click = float(self.mask_threshold_var.get())
        continuous_at_click = bool(self.continuous_mask_only_var.get())
        model = self.model

        def worker() -> None:
            try:
                points = np.asarray([seed_point], dtype=np.float32)
                labels = np.asarray([1], dtype=np.int32)
                mask = model.predict_mask_from_points(points=points, point_labels=labels)
                mask = self._stabilize_mask_for_seed(
                    mask,
                    seed_point=seed_point,
                    trim_component=continuous_at_click,
                )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda: self._finish_ignore_prediction(
                        seed_point,
                        threshold_at_click,
                        continuous_at_click,
                        None,
                        str(exc),
                    ),
                )
                return
            self.root.after(
                0,
                lambda: self._finish_ignore_prediction(
                    seed_point,
                    threshold_at_click,
                    continuous_at_click,
                    mask,
                    None,
                ),
            )

        self.status_var.set("Adding ignored region...")
        threading.Thread(target=worker, daemon=True).start()

    def _finish_ignore_prediction(
        self,
        seed_point: tuple[float, float],
        threshold_at_click: float,
        continuous_at_click: bool,
        mask: np.ndarray | None,
        err: str | None,
    ) -> None:
        with self._predict_lock:
            self._predict_busy = False
        if err is not None:
            self.status_var.set(f"Ignore warning: {err}")
        elif mask is not None:
            self.ignore_masks.append(mask.copy())
            self._ignore_mask_metadata.append(
                self._build_ignore_mask_metadata(
                    ignore_index=len(self.ignore_masks) - 1,
                    seed_point=seed_point,
                    mask=mask,
                    threshold=threshold_at_click,
                    continuous_mask_only=continuous_at_click,
                )
            )
            self._invalidate_committed_overlay_cache()
            self._save_annotations_for_current_image()
            self._render()
            self.status_var.set(f"Added ignored region {len(self.ignore_masks)}.")
        if self._pending_navigation_delta is not None:
            self._dirty_preview = False
            self.hover_point = None
            self._hover_preview_mask = None
            self.root.after(0, self._consume_pending_navigation)
        elif self._dirty_preview:
            self.root.after(8, self._run_preview)

    def _build_ignore_mask_metadata(
        self,
        ignore_index: int,
        seed_point: tuple[float, float],
        mask: np.ndarray,
        threshold: float,
        continuous_mask_only: bool,
    ) -> dict[str, Any]:
        h, w = self.image_np.shape[:2]
        return mask_engine.build_ignore_mask_metadata(
            ignore_index=ignore_index,
            seed_point=seed_point,
            mask=mask,
            threshold=threshold,
            continuous_mask_only=continuous_mask_only,
            saved_at=self._utc_now_iso(),
            image_shape_hw=(int(h), int(w)),
            model_name=self.model_name_var.get().strip() or DEFAULT_MODEL_NAME,
        )
    def _display_radius_to_image_radius(self, radius_display_px: float) -> float:
        if self.image_np is None:
            return float(radius_display_px)
        iw = float(self.image_np.shape[1])
        ih = float(self.image_np.shape[0])
        dw, dh = self.display_size
        if dw <= 1 or dh <= 1:
            return float(radius_display_px)
        sx = iw / float(dw)
        sy = ih / float(dh)
        return float(radius_display_px) * (sx + sy) * 0.5

    def _remove_committed_masks_near_image(
        self, image_center: tuple[float, float], radius_image_px: float
    ) -> int:
        if not self.committed_masks:
            return 0
        fallback_threshold = float(self.mask_threshold_var.get())
        cx = int(round(float(image_center[0])))
        cy = int(round(float(image_center[1])))
        if self.image_np is None:
            return 0
        h, w = self.image_np.shape[:2]
        cx = int(np.clip(cx, 0, max(0, w - 1)))
        cy = int(np.clip(cy, 0, max(0, h - 1)))

        r = max(1, int(round(float(radius_image_px))))
        x0 = max(0, cx - r)
        x1 = min(w, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(h, cy + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= float(r * r)

        removed = 0
        # Iterate backwards so index removals are stable.
        for idx in range(len(self.committed_masks) - 1, -1, -1):
            mask = self._binary_committed_mask_at_index(idx, fallback_threshold)
            region = mask[y0:y1, x0:x1]
            if region.shape != disk.shape:
                continue
            if bool(np.any(region[disk])):
                if self._remove_committed_mask_at_index(idx):
                    removed += 1
        return removed

    def _remove_ignore_masks_near_image(
        self, image_center: tuple[float, float], radius_image_px: float
    ) -> int:
        if not getattr(self, "ignore_masks", []):
            return 0
        fallback_threshold = float(self.mask_threshold_var.get())
        cx = int(round(float(image_center[0])))
        cy = int(round(float(image_center[1])))
        if self.image_np is None:
            return 0
        h, w = self.image_np.shape[:2]
        cx = int(np.clip(cx, 0, max(0, w - 1)))
        cy = int(np.clip(cy, 0, max(0, h - 1)))

        r = max(1, int(round(float(radius_image_px))))
        x0 = max(0, cx - r)
        x1 = min(w, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(h, cy + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= float(r * r)

        removed = 0
        for idx in range(len(self.ignore_masks) - 1, -1, -1):
            mask = self._binary_ignore_mask_at_index(idx, fallback_threshold)
            region = mask[y0:y1, x0:x1]
            if region.shape != disk.shape:
                continue
            if bool(np.any(region[disk])):
                if self._remove_ignore_mask_at_index(idx):
                    removed += 1
        return removed

    def _remove_ignore_mask_at_index(self, idx: int) -> bool:
        if idx < 0 or idx >= len(getattr(self, "ignore_masks", [])):
            return False
        self.ignore_masks.pop(idx)
        if idx < len(self._ignore_mask_metadata):
            self._ignore_mask_metadata.pop(idx)
        for new_idx, md in enumerate(self._ignore_mask_metadata):
            md["ignore_index"] = int(new_idx)
        self._invalidate_committed_overlay_cache()
        return True

    def _on_mouse_move(self, event: tk.Event) -> None:
        if self._middle_pan_last_canvas_xy is not None:
            return
        p = self._to_image_coords(event.x, event.y)
        if p is None:
            if self.hover_point is not None or self._hover_preview_mask is not None:
                self.hover_point = None
                self._hover_preview_mask = None
                self._render()
            return
        self.hover_point = p
        self._hover_preview_mask = None
        self._schedule_preview()

    def _on_mouse_leave(self, _event: tk.Event) -> None:
        self.hover_point = None
        self._hover_preview_mask = None
        self._render()

    def _clear_prompts(self, reset_mask: bool = True, persist: bool = True) -> None:
        self._dirty_preview = False
        self._request_id += 1
        self._pending_navigation_delta = None
        self.pos_points.clear()
        self.neg_points.clear()
        self._seed_mask_cache.clear()
        self._auto_advance_session_id = None
        self.hover_point = None
        self._hover_preview_mask = None
        if reset_mask:
            self.last_mask = None
            self.committed_masks.clear()
            self.ignore_masks.clear()
            self._ignore_mask_metadata.clear()
            self._invalidate_committed_overlay_cache()
            self._session_to_mask_index.clear()
            self._session_metadata.clear()
            self._active_session_id = 0
            if persist:
                self._save_annotations_for_current_image()
        self._render()
        self.status_var.set("Prompts cleared.")

    def _schedule_preview(self) -> None:
        if self.model is None or self.image_np is None:
            self._render()
            return
        self._dirty_preview = True
        if not self._predict_busy:
            self.root.after(12, self._run_preview)

    def _run_preview(self) -> None:
        if not self._dirty_preview:
            return
        if self.model is None or self.image_np is None:
            return

        with self._predict_lock:
            if self._predict_busy:
                return
            self._predict_busy = True
            self._dirty_preview = False
            self._request_id += 1
            req_id = self._request_id

        pos = list(self.pos_points)
        neg = list(self.neg_points)
        hover_only_preview = self.hover_point is not None and not bool(self.pos_points)
        # Once a user has clicked, treat those prompts as committed seeds and do
        # not keep mutating them with hover-only preview points.
        if self.hover_point is not None and not pos:
            pos.append(self.hover_point)

        if not pos:
            self.last_mask = None
            self._hover_preview_mask = None
            self._seed_mask_cache.clear()
            session_id = int(self._active_session_id)
            if session_id > 0:
                had_mask = session_id in self._session_to_mask_index
                self._remove_committed_mask_for_session(session_id)
                if had_mask:
                    self._save_annotations_for_current_image()
            self._predict_busy = False
            self._render()
            return

        pos_points = [(float(x), float(y)) for x, y in pos]
        neg_points = [(float(x), float(y)) for x, y in neg]
        continuous_mask_only = bool(self.continuous_mask_only_var.get())
        session_id = int(self._active_session_id)

        model = self.model
        if model is None:
            with self._predict_lock:
                self._predict_busy = False
            return

        def worker() -> None:
            try:
                apply_stabilization = not hover_only_preview
                cacheable = (len(neg_points) == 0) and apply_stabilization
                if not cacheable:
                    self._seed_mask_cache.clear()
                trim_component = apply_stabilization and continuous_mask_only
                combined_mask: np.ndarray | None = None
                for px, py in pos_points:
                    key = (int(round(px)), int(round(py)), bool(continuous_mask_only))
                    current_mask = self._seed_mask_cache.get(key) if cacheable else None
                    if current_mask is None:
                        points = np.asarray([(px, py), *neg_points], dtype=np.float32)
                        labels = np.asarray([1] + [0] * len(neg_points), dtype=np.int32)
                        current_mask = model.predict_mask_from_points(
                            points=points, point_labels=labels
                        )
                        if apply_stabilization:
                            current_mask = self._stabilize_mask_for_seed(
                                current_mask,
                                seed_point=(px, py),
                                trim_component=trim_component,
                            )
                        if cacheable:
                            self._seed_mask_cache[key] = current_mask
                    if combined_mask is None:
                        combined_mask = current_mask
                    else:
                        combined_mask = np.maximum(combined_mask, current_mask)
            except Exception as exc:
                self.root.after(
                    0,
                    lambda: self._finish_preview(
                        req_id,
                        session_id,
                        None,
                        str(exc),
                        hover_only_preview,
                    ),
                )
                return
            self.root.after(
                0,
                lambda: self._finish_preview(
                    req_id,
                    session_id,
                    combined_mask,
                    None,
                    hover_only_preview,
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _stabilize_mask_for_seed(
        self,
        mask_logits: np.ndarray,
        seed_point: tuple[float, float],
        trim_component: bool = True,
    ) -> np.ndarray:
        """Prefer the connected foreground region around the clicked seed point."""
        mask = np.asarray(mask_logits, dtype=np.float32)
        if mask.ndim != 2:
            return mask

        h, w = mask.shape
        sx = int(np.clip(round(seed_point[0]), 0, max(0, w - 1)))
        sy = int(np.clip(round(seed_point[1]), 0, max(0, h - 1)))

        # If the model chose the opposite side, invert logits so seed is foreground.
        if float(mask[sy, sx]) <= 0.0:
            mask = -mask

        if not trim_component:
            return mask

        foreground = mask > 0.0
        if not bool(foreground[sy, sx]):
            return mask

        keep = self._connected_component_from_seed(foreground, sx, sy)
        if keep is None:
            return mask

        out = np.full_like(mask, fill_value=float(mask.min()) - 1.0)
        out[keep] = mask[keep]
        return out

    @staticmethod
    def _connected_component_from_seed(
        foreground: np.ndarray, sx: int, sy: int
    ) -> np.ndarray | None:
        h, w = foreground.shape
        if not bool(foreground[sy, sx]):
            return None

        visited = np.zeros((h, w), dtype=bool)
        q: deque[tuple[int, int]] = deque()
        q.append((sx, sy))
        visited[sy, sx] = True

        while q:
            x, y = q.popleft()
            for nx, ny in (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
                (x - 1, y - 1),
                (x + 1, y - 1),
                (x - 1, y + 1),
                (x + 1, y + 1),
            ):
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                if visited[ny, nx] or (not bool(foreground[ny, nx])):
                    continue
                visited[ny, nx] = True
                q.append((nx, ny))
        return visited

    def _finish_preview(
        self,
        req_id: int,
        session_id: int,
        mask: np.ndarray | None,
        err: str | None,
        is_hover_preview: bool = False,
    ) -> None:
        with self._predict_lock:
            self._predict_busy = False
        if req_id != self._request_id:
            # stale result
            return
        should_auto_advance = False
        if err is not None:
            self.status_var.set(f"Prediction warning: {err}")
            if self._auto_advance_session_id == int(session_id):
                self._auto_advance_session_id = None
            if is_hover_preview:
                self._hover_preview_mask = None
        else:
            if is_hover_preview:
                self.last_mask = None
                self._hover_preview_mask = mask.copy() if mask is not None else None
            else:
                self._hover_preview_mask = None
                self.last_mask = mask
                if mask is not None and session_id > 0:
                    existing = self._session_to_mask_index.get(session_id)
                    if existing is None:
                        self.committed_masks.append(mask.copy())
                        self._session_to_mask_index[session_id] = len(self.committed_masks) - 1
                    else:
                        self.committed_masks[existing] = mask.copy()
                    self._invalidate_committed_overlay_cache()
                    self._update_session_metadata_with_mask(session_id=session_id, mask=mask)
                    self._save_annotations_for_current_image()
                    should_auto_advance = self._auto_advance_session_id == int(session_id)
                elif self._auto_advance_session_id == int(session_id):
                    self._auto_advance_session_id = None
        if should_auto_advance:
            self._start_new_mask_session()
        elif self._pending_navigation_delta is None:
            self._render()

        if self._pending_navigation_delta is not None:
            self._dirty_preview = False
            self.hover_point = None
            self._hover_preview_mask = None
            self.root.after(0, self._consume_pending_navigation)
        elif self._dirty_preview:
            self.root.after(8, self._run_preview)

    def _remove_committed_mask_for_session(self, session_id: int) -> None:
        sid = int(session_id)
        idx = self._session_to_mask_index.pop(sid, None)
        self._session_metadata.pop(sid, None)
        if idx is None:
            return
        self._remove_committed_mask_at_index(idx)

    def _remove_committed_mask_at_index(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self.committed_masks):
            return False
        self.committed_masks.pop(idx)
        self._invalidate_committed_overlay_cache()

        for sid, cur_idx in list(self._session_to_mask_index.items()):
            if cur_idx == idx:
                self._session_to_mask_index.pop(sid, None)
                self._session_metadata.pop(int(sid), None)
            elif cur_idx > idx:
                self._session_to_mask_index[sid] = cur_idx - 1
        return True

    def _get_committed_click_points_in_order(
        self,
    ) -> list[tuple[int, tuple[float, float]]]:
        """Return numbered click seed points in current mask display order (1..N)."""
        index_to_session: dict[int, int] = {}
        for sid, idx in self._session_to_mask_index.items():
            if 0 <= int(idx) < len(self.committed_masks):
                index_to_session[int(idx)] = int(sid)

        points: list[tuple[int, tuple[float, float]]] = []
        for idx in range(len(self.committed_masks)):
            sid = index_to_session.get(idx)
            if sid is None:
                continue
            md = self._session_metadata.get(int(sid), {})
            seed = md.get("seed_point_xy")
            if isinstance(seed, (list, tuple)) and len(seed) >= 2:
                try:
                    px = float(seed[0])
                    py = float(seed[1])
                    points.append((idx + 1, (px, py)))
                except Exception:
                    continue
        return points

    def _get_ignore_click_points_in_order(
        self,
    ) -> list[tuple[int, tuple[float, float]]]:
        points: list[tuple[int, tuple[float, float]]] = []
        for idx, md in enumerate(getattr(self, "_ignore_mask_metadata", [])):
            seed = md.get("seed_point_xy")
            if isinstance(seed, (list, tuple)) and len(seed) >= 2:
                try:
                    px = float(seed[0])
                    py = float(seed[1])
                    points.append((idx + 1, (px, py)))
                except Exception:
                    continue
        return points

    def _render(self) -> None:
        if self.display_base_np is None:
            self.canvas.delete("all")
            return

        self._clamp_view()
        base = self.display_base_np
        dw, dh = self.display_size
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        x0 = int(self.view_x)
        y0 = int(self.view_y)
        x1 = min(dw, x0 + canvas_w)
        y1 = min(dh, y0 + canvas_h)
        frame = base[y0:y1, x0:x1].copy()

        threshold = float(self.mask_threshold_var.get())
        crop_box = (x0, y0, x1, y1)
        self._blend_committed_overlay_on_frame(frame, threshold, crop_box)
        if not self.committed_masks and self.last_mask is not None:
            self._overlay_mask_on_frame(
                frame,
                self.last_mask,
                threshold,
                crop_box,
                self._mask_color_for_index(0),
                MASK_OVERLAY_ALPHA,
            )
        if self._hover_preview_mask is not None:
            self._overlay_mask_on_frame(
                frame,
                self._hover_preview_mask,
                threshold,
                crop_box,
                self._mask_color_for_index(len(self.committed_masks)),
                HOVER_MASK_OVERLAY_ALPHA,
            )

        outline_mask = self._active_selection_mask_for_outline()
        if bool(self.selection_outline_var.get()) and outline_mask is not None:
            self._outline_mask_on_frame(
                frame,
                outline_mask,
                threshold,
                crop_box,
                self._outline_color_for_name(self.selection_outline_color_var.get()),
            )

        # Draw prompt points (in display coordinates)
        iw = self.image_np.shape[1] if self.image_np is not None else 1
        ih = self.image_np.shape[0] if self.image_np is not None else 1
        dw, dh = self.display_size

        def to_disp(pt: tuple[float, float]) -> tuple[int, int]:
            x_full = pt[0] * dw / max(1, iw)
            y_full = pt[1] * dh / max(1, ih)
            x = int(round(x_full - self.view_x))
            y = int(round(y_full - self.view_y))
            return x, y

        for pt in self.pos_points:
            x, y = to_disp(pt)
            self._draw_dot(frame, x, y, color=(0, 255, 0), radius=2)
        for pt in self.neg_points:
            x, y = to_disp(pt)
            self._draw_dot(frame, x, y, color=(255, 0, 0), radius=2)
        if self.hover_point is not None:
            x, y = to_disp(self.hover_point)
            self._draw_dot(frame, x, y, color=(255, 255, 0), radius=4)

        img = Image.fromarray(frame)
        self.display_photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_photo)

        # Draw click indices (1..N) at committed seed points.
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        for num, pt in self._get_committed_click_points_in_order():
            canvas_pt = self._image_point_to_canvas(pt)
            if canvas_pt is None:
                continue
            cx, cy = canvas_pt
            if cx < 0 or cy < 0 or cx >= canvas_w or cy >= canvas_h:
                continue
            x = int(round(cx))
            y = int(round(cy))
            label_fill = self._rgb_to_hex(self._mask_color_for_index(num - 1))
            # Light outline for readability across bright/dark backgrounds.
            for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                self.canvas.create_text(
                    x + ox,
                    y + oy,
                    text=str(num),
                    fill="#111111",
                    font=("TkDefaultFont", 11, "bold"),
                    anchor=tk.CENTER,
                )
            self.canvas.create_text(
                x,
                y,
                text=str(num),
                fill=label_fill,
                font=("TkDefaultFont", 11, "bold"),
                anchor=tk.CENTER,
            )

        for _num, pt in self._get_ignore_click_points_in_order():
            canvas_pt = self._image_point_to_canvas(pt)
            if canvas_pt is None:
                continue
            cx, cy = canvas_pt
            if cx < 0 or cy < 0 or cx >= canvas_w or cy >= canvas_h:
                continue
            x = int(round(cx))
            y = int(round(cy))
            for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                self.canvas.create_text(
                    x + ox,
                    y + oy,
                    text="X",
                    fill="#111111",
                    font=("TkDefaultFont", 11, "bold"),
                    anchor=tk.CENTER,
                )
            self.canvas.create_text(
                x,
                y,
                text="X",
                fill=IGNORE_MASK_LABEL_COLOR,
                font=("TkDefaultFont", 11, "bold"),
                anchor=tk.CENTER,
            )

    def _invalidate_committed_overlay_cache(self) -> None:
        self._committed_overlay_signature = None
        self._committed_overlay_rgb = None
        self._committed_overlay_base_weight = None

    def _committed_overlay_cache_signature(
        self,
        thresholds: list[float],
        ignore_thresholds: list[float],
    ) -> tuple[Any, ...]:
        return (
            self.display_size,
            tuple(float(value) for value in thresholds),
            tuple(float(value) for value in ignore_thresholds),
            tuple(id(mask) for mask in self.committed_masks),
            tuple(id(mask) for mask in getattr(self, "ignore_masks", [])),
        )

    def _ensure_committed_overlay_cache(self, threshold: float) -> None:
        thresholds = self._thresholds_for_committed_masks(threshold)
        ignore_thresholds = self._thresholds_for_ignore_masks(threshold)
        signature = self._committed_overlay_cache_signature(thresholds, ignore_thresholds)
        if (
            self._committed_overlay_signature == signature
            and self._committed_overlay_rgb is not None
            and self._committed_overlay_base_weight is not None
        ):
            return

        if not self.committed_masks and not getattr(self, "ignore_masks", []):
            self._committed_overlay_signature = signature
            self._committed_overlay_rgb = None
            self._committed_overlay_base_weight = None
            return

        dw, dh = self.display_size
        overlay_rgb = np.zeros((dh, dw, 3), dtype=np.float32)
        base_weight = np.ones((dh, dw), dtype=np.float32)
        alpha = float(MASK_OVERLAY_ALPHA)
        keep_base = 1.0 - alpha
        for idx in range(len(self.committed_masks)):
            mask_arr = self._binary_committed_mask_at_index(idx, threshold)
            if mask_arr.ndim != 2:
                continue
            mask_img = Image.fromarray((mask_arr.astype(np.uint8) * 255), mode="L")
            mask_img = mask_img.resize(self.display_size, Image.Resampling.NEAREST)
            mask_display = np.asarray(mask_img) > 0
            if not bool(np.any(mask_display)):
                continue
            color_arr = np.asarray(self._mask_color_for_index(idx), dtype=np.float32)
            overlay_rgb[mask_display] = keep_base * overlay_rgb[mask_display] + alpha * color_arr
            base_weight[mask_display] *= keep_base

        ignore_alpha = float(IGNORE_MASK_OVERLAY_ALPHA)
        ignore_keep_base = 1.0 - ignore_alpha
        ignore_color_arr = np.asarray(IGNORE_MASK_OVERLAY_COLOR, dtype=np.float32)
        for idx in range(len(getattr(self, "ignore_masks", []))):
            mask_arr = self._binary_ignore_mask_at_index(idx, threshold)
            if mask_arr.ndim != 2:
                continue
            mask_img = Image.fromarray((mask_arr.astype(np.uint8) * 255), mode="L")
            mask_img = mask_img.resize(self.display_size, Image.Resampling.NEAREST)
            mask_display = np.asarray(mask_img) > 0
            if not bool(np.any(mask_display)):
                continue
            overlay_rgb[mask_display] = (
                ignore_keep_base * overlay_rgb[mask_display]
                + ignore_alpha * ignore_color_arr
            )
            base_weight[mask_display] *= ignore_keep_base

        self._committed_overlay_signature = signature
        self._committed_overlay_rgb = overlay_rgb
        self._committed_overlay_base_weight = base_weight

    def _blend_committed_overlay_on_frame(
        self,
        frame: np.ndarray,
        threshold: float,
        crop_box: tuple[int, int, int, int],
    ) -> None:
        self._ensure_committed_overlay_cache(threshold)
        if self._committed_overlay_rgb is None or self._committed_overlay_base_weight is None:
            return
        x0, y0, x1, y1 = crop_box
        rgb_crop = self._committed_overlay_rgb[y0:y1, x0:x1]
        weight_crop = self._committed_overlay_base_weight[y0:y1, x0:x1]
        active = weight_crop < 0.999
        if not bool(np.any(active)):
            return
        frame[active] = (
            frame[active].astype(np.float32) * weight_crop[active, None]
            + rgb_crop[active]
        ).astype(np.uint8)

    @staticmethod
    def _mask_color_for_index(index: int) -> tuple[int, int, int]:
        return MASK_OVERLAY_COLORS[int(index) % len(MASK_OVERLAY_COLORS)]

    @staticmethod
    def _normalize_outline_color_name(raw_name: str) -> str | None:
        wanted = str(raw_name or "").strip().lower().replace("_", " ").replace("-", " ")
        for name, _color in OUTLINE_COLOR_CHOICES:
            normalized = name.lower().replace("_", " ").replace("-", " ")
            if wanted in {normalized, normalized.removeprefix("neon ")}:
                return name
        return None

    @staticmethod
    def _outline_color_for_name(raw_name: str) -> tuple[int, int, int]:
        name = SamHoverMaskApp._normalize_outline_color_name(raw_name)
        if name is None:
            name = DEFAULT_OUTLINE_COLOR_NAME
        return OUTLINE_COLOR_MAP[name]

    @staticmethod
    def _rgb_to_hex(color: tuple[int, int, int]) -> str:
        return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

    def _active_selection_mask_for_outline(self) -> np.ndarray | None:
        if self._hover_preview_mask is not None:
            return self._hover_preview_mask
        if self.pos_points and self.last_mask is not None:
            return self.last_mask
        return None

    def _overlay_mask_on_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        threshold: float,
        crop_box: tuple[int, int, int, int],
        color: tuple[int, int, int],
        alpha: float,
        subtract_ignored: bool = True,
    ) -> None:
        mask_arr = self._binary_mask_with_ignore(mask, threshold, subtract_ignored=subtract_ignored)
        if mask_arr.ndim != 2:
            return
        mask_img = Image.fromarray((mask_arr.astype(np.uint8) * 255), mode="L")
        mask_img = mask_img.resize(self.display_size, Image.Resampling.NEAREST)
        mask_crop = mask_img.crop(crop_box)
        mask_crop_arr = np.asarray(mask_crop) > 0
        if not bool(np.any(mask_crop_arr)):
            return
        color_arr = np.asarray(color, dtype=np.float32)
        frame[mask_crop_arr] = (
            (1.0 - float(alpha)) * frame[mask_crop_arr].astype(np.float32)
            + float(alpha) * color_arr
        ).astype(np.uint8)

    def _outline_mask_on_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        threshold: float,
        crop_box: tuple[int, int, int, int],
        color: tuple[int, int, int],
        subtract_ignored: bool = True,
    ) -> None:
        mask_arr = self._binary_mask_with_ignore(mask, threshold, subtract_ignored=subtract_ignored)
        if mask_arr.ndim != 2:
            return
        mask_img = Image.fromarray((mask_arr.astype(np.uint8) * 255), mode="L")
        mask_img = mask_img.resize(self.display_size, Image.Resampling.NEAREST)
        mask_crop = mask_img.crop(crop_box)
        mask_crop_arr = np.asarray(mask_crop) > 0
        if not bool(np.any(mask_crop_arr)):
            return

        padded = np.pad(mask_crop_arr, 1, mode="constant", constant_values=False)
        neighbors_all_foreground = (
            padded[1:-1, :-2]
            & padded[1:-1, 2:]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
        )
        outline = mask_crop_arr & (~neighbors_all_foreground)
        if not bool(np.any(outline)):
            return
        frame[outline] = np.asarray(color, dtype=np.uint8)

    @staticmethod
    def _draw_dot(frame: np.ndarray, x: int, y: int, color: tuple[int, int, int], radius: int = 5) -> None:
        h, w = frame.shape[:2]
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius
        frame[y0:y1, x0:x1][disk] = color


def launch_gui(initial_source_input: Path | None, params: dict[str, Any]) -> int:
    root = tk.Tk()
    _ = SamHoverMaskApp(
        root,
        initial_source_input=initial_source_input,
        initial_params=params,
    )
    root.mainloop()
    return 0


def run_stage_for_runner(source_input: Path, params: dict[str, Any]) -> dict[str, Any]:
    if not source_input.exists():
        raise RuntimeError(f"Source input does not exist: {source_input}")
    if source_input.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise RuntimeError(
            "Unsupported source input extension for this stage: "
            f"{source_input.suffix} (expected one of {sorted(SUPPORTED_IMAGE_EXTENSIONS)})"
        )

    try:
        image_np = np.asarray(Image.open(source_input).convert("RGB"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load source image: {source_input}") from exc

    model_name = str(params.get("model_name", DEFAULT_MODEL_NAME)).strip() or DEFAULT_MODEL_NAME
    encoder_raw = str(params.get("encoder_path", "")).strip()
    decoder_raw = str(params.get("decoder_path", "")).strip()

    if bool(encoder_raw) ^ bool(decoder_raw):
        raise RuntimeError("Provide both 'encoder_path' and 'decoder_path', or neither.")

    if encoder_raw and decoder_raw:
        encoder_path = Path(encoder_raw).expanduser().resolve()
        decoder_path = Path(decoder_raw).expanduser().resolve()
        if not encoder_path.exists() or not decoder_path.exists():
            raise RuntimeError(
                "Custom encoder/decoder paths do not exist: "
                f"{encoder_path} / {decoder_path}"
            )
    else:
        encoder_path, decoder_path = model_weight_paths(model_name, MODELS_DIR)
        if not encoder_path.exists() or not decoder_path.exists():
            encoder_path, decoder_path = download_selected_model(
                model_name=model_name,
                models_dir=MODELS_DIR,
            )

    model = SegmentAnythingOnnx(str(encoder_path), str(decoder_path))
    model.set_image(image_np)
    embedding = model.get_embedding()
    if embedding is None:
        raise RuntimeError("Model did not produce an image embedding.")

    model_cache_token = build_model_cache_token(
        model_name=model_name,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        image_size=int(model.image_size),
    )
    model_cache_metadata = build_model_cache_metadata(
        model_name=model_name,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        image_size=int(model.image_size),
    )
    embedding_cache_path = save_embedding_cache_for_image(
        image_path=source_input,
        image_np=image_np,
        embedding=embedding,
        model_cache_token=model_cache_token,
        model_metadata=model_cache_metadata,
    )

    annotation_path = annotation_store_path_for_image(source_input)
    annotation_metadata_path = annotation_metadata_path_for_image(source_input)
    annotation_metadata = load_json_object_from_path_optional(annotation_metadata_path)
    saved_mask_count = count_saved_masks(annotation_path)
    return {
        "action": "process",
        "source_input": str(source_input),
        "source_name": source_input.name,
        "source_size_hw": [int(image_np.shape[0]), int(image_np.shape[1])],
        "model_name": model_name,
        "encoder_path": str(encoder_path),
        "decoder_path": str(decoder_path),
        "embedding_cache_path": str(embedding_cache_path),
        "embedding_cache_exists": embedding_cache_path.exists(),
        "annotation_path": str(annotation_path),
        "annotation_exists": annotation_path.exists(),
        "annotation_metadata_path": str(annotation_metadata_path),
        "annotation_metadata_exists": annotation_metadata_path.exists(),
        "saved_mask_count": int(saved_mask_count),
        "annotation_mask_count": int(annotation_metadata.get("mask_count", 0))
        if annotation_metadata
        else int(saved_mask_count),
        "annotation_total_mask_area_px": int(annotation_metadata.get("total_mask_area_px", 0))
        if annotation_metadata
        else 0,
    }


def main() -> int:
    args = parse_args()

    has_source = has_text(args.source_input)
    has_output = has_text(args.output_result_json)
    has_params = has_text(args.params_json)

    full_runner_mode = has_source and has_output and has_params
    partial_runner_mode = (has_source or has_output) and not full_runner_mode

    if args.gui or not full_runner_mode:
        if partial_runner_mode and not args.gui:
            raise RuntimeError(
                "Partial runner args detected. Provide all of "
                "--source-input, --output-result-json, --params-json "
                "or run with no runner args for GUI mode."
            )

        gui_params_raw = args.gui_params_json or args.params_json or ""
        gui_params = load_json_object_optional(gui_params_raw)
        gui_source_raw = args.gui_source_input or args.source_input or ""
        gui_source = (
            Path(gui_source_raw).expanduser().resolve()
            if has_text(gui_source_raw)
            else resolve_preload_source_input(gui_params)
        )
        return launch_gui(gui_source, gui_params)

    source_input = Path(str(args.source_input)).expanduser().resolve()
    output_result_json = Path(str(args.output_result_json)).expanduser().resolve()
    params_path = Path(str(args.params_json)).expanduser().resolve()
    params = load_json_object(params_path)

    run_mode = resolve_run_mode()
    open_results = as_bool(params, "open_results", False)

    if run_mode == "visual" or open_results:
        launch_gui(source_input, params)
        annotation_path = annotation_store_path_for_image(source_input)
        annotation_metadata_path = annotation_metadata_path_for_image(source_input)
        annotation_metadata = load_json_object_from_path_optional(annotation_metadata_path)
        payload: dict[str, Any] = {
            "action": "open_results",
            "source_input": str(source_input),
            "annotation_path": str(annotation_path),
            "annotation_exists": annotation_path.exists(),
            "annotation_metadata_path": str(annotation_metadata_path),
            "annotation_metadata_exists": annotation_metadata_path.exists(),
            "saved_mask_count": count_saved_masks(annotation_path),
            "annotation_mask_count": int(annotation_metadata.get("mask_count", 0))
            if annotation_metadata
            else count_saved_masks(annotation_path),
            "annotation_total_mask_area_px": int(
                annotation_metadata.get("total_mask_area_px", 0)
            )
            if annotation_metadata
            else 0,
        }
    else:
        payload = run_stage_for_runner(source_input, params)

    payload.setdefault("source_input", str(source_input))
    payload.setdefault("source_exists", source_input.exists())
    payload.setdefault("run_mode", run_mode)
    payload.setdefault("open_results", open_results)

    output_result_json.parent.mkdir(parents=True, exist_ok=True)
    output_result_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote result JSON: {output_result_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
