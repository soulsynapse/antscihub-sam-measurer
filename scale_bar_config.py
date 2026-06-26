#!/usr/bin/env python3
from __future__ import annotations

import argparse
from version import __version__
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IMAGE_GLOBS_DEFAULT = "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SCALE_BAR_CONFIG_SUFFIX = ".scale_bar_config.result.json"
APPLIES_TO_PREVIEW_LIMIT_DEFAULT = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Configure one scale bar calibration from a reference image and "
            "apply it to all images in the source folder."
        ),
    )
    parser.add_argument("--source-input", "--source-folder", dest="source_input")
    parser.add_argument(
        "--output-result-json",
        "--output-scale-bar-config-json",
        dest="output_result_json",
    )
    parser.add_argument("--params-json")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Force direct GUI mode.",
    )
    parser.add_argument(
        "--gui-source-input",
        default="",
        help="Optional source folder/image when launching GUI directly.",
    )
    parser.add_argument(
        "--gui-params-json",
        default="",
        help="Optional params JSON path for GUI preload.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args()


def has_text(raw: str | None) -> bool:
    return isinstance(raw, str) and bool(raw.strip())


def bring_tk_window_to_front(window: Any) -> None:
    try:
        window.deiconify()
        window.lift()
        window.attributes("-topmost", True)
        window.focus_force()
        window.after(600, lambda: window.attributes("-topmost", False))
    except Exception:
        pass


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
    raw = params.get(key, default)
    parsed = parse_bool_like(raw)
    if parsed is not None:
        return parsed
    return bool(default)


def as_float(params: dict[str, Any], key: str, default: float) -> float:
    raw = params.get(key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def as_int(params: dict[str, Any], key: str, default: int) -> int:
    raw = params.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def as_str(params: dict[str, Any], key: str, default: str) -> str:
    raw = params.get(key, default)
    if isinstance(raw, str):
        return raw
    return str(default)


def resolve_run_mode() -> str:
    run_mode = os.environ.get("PIPEYARD_RUN_MODE", "").strip().lower()
    if run_mode in {"visual", "headless"}:
        return run_mode

    visual = parse_bool_like(os.environ.get("PIPEYARD_VISUAL_MODE"))
    if visual is True:
        return "visual"

    headless = parse_bool_like(os.environ.get("PIPEYARD_HEADLESS"))
    if headless is True:
        return "headless"

    return "headless"


def load_json_object(params_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(params_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read JSON file: {params_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON file must contain an object: {params_path}")
    return payload


def load_json_object_optional(raw_path: str) -> dict[str, Any]:
    text = str(raw_path or "").strip()
    if not text:
        return {}
    path = Path(text).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"JSON file does not exist: {path}")
    return load_json_object(path)


def require_pillow_image() -> Any:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "This stage requires Pillow. Install dependency: Pillow>=10.4.0"
        ) from exc
    return Image


def parse_glob_patterns(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        text = IMAGE_GLOBS_DEFAULT

    pieces = [piece.strip() for piece in text.replace(",", ";").split(";") if piece.strip()]
    return pieces or IMAGE_GLOBS_DEFAULT.split(";")


def collect_image_candidates(source_dir: Path, patterns: list[str]) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    for pattern in patterns:
        for path in sorted(source_dir.glob(pattern)):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(path.resolve())

    if candidates:
        return candidates

    for path in sorted(source_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            candidates.append(path.resolve())
    return candidates


def find_existing_scale_bar_configs(source_dir: Path) -> list[Path]:
    return sorted(
        (
            path.resolve()
            for path in source_dir.glob(f"*{SCALE_BAR_CONFIG_SUFFIX}")
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def image_path_from_existing_scale_bar_config(
    config_path: Path,
    source_dir: Path,
    available_images: list[Path],
) -> Path | None:
    try:
        payload = load_json_object(config_path)
    except Exception:
        return None

    image_by_key = {str(path.resolve()).lower(): path.resolve() for path in available_images}
    image_by_name = {path.name: path.resolve() for path in available_images}
    image_by_stem = {path.stem: path.resolve() for path in available_images}

    selected_image_path = str(payload.get("selected_image_path") or "").strip()
    if selected_image_path:
        try:
            candidate = Path(selected_image_path).expanduser().resolve()
        except Exception:
            candidate = source_dir / selected_image_path
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            matched_image = image_by_key.get(str(candidate.resolve()).lower())
            if matched_image is not None:
                return matched_image

    selected_image_name = str(payload.get("selected_image_name") or "").strip()
    if selected_image_name and selected_image_name in image_by_name:
        return image_by_name[selected_image_name]

    suffix_len = len(SCALE_BAR_CONFIG_SUFFIX)
    if config_path.name.endswith(SCALE_BAR_CONFIG_SUFFIX):
        config_stem = config_path.name[:-suffix_len]
        if config_stem in image_by_stem:
            return image_by_stem[config_stem]

    return None


def find_existing_scale_bar_config_for_image(
    source_dir: Path,
    image_path: Path,
    available_images: list[Path],
) -> Path | None:
    image_key = str(image_path.resolve()).lower()
    direct_candidate = image_path.with_name(f"{image_path.stem}{SCALE_BAR_CONFIG_SUFFIX}")
    if direct_candidate.is_file():
        return direct_candidate.resolve()

    for config_path in find_existing_scale_bar_configs(source_dir):
        config_image = image_path_from_existing_scale_bar_config(
            config_path,
            source_dir,
            available_images,
        )
        if config_image is not None and str(config_image.resolve()).lower() == image_key:
            return config_path
    return None


def resolve_reference_and_applicability(
    source_input: Path,
    params: dict[str, Any],
) -> tuple[Path, list[Path], Path | None, Path | None]:
    patterns = parse_glob_patterns(as_str(params, "image_glob", IMAGE_GLOBS_DEFAULT))

    if source_input.is_file():
        if source_input.suffix.lower() not in IMAGE_SUFFIXES:
            raise RuntimeError(
                f"Source file must be an image ({', '.join(sorted(IMAGE_SUFFIXES))}): {source_input}"
            )
        resolved_file = source_input.resolve()
        folder_path = resolved_file.parent.resolve()
        applicable_images = collect_image_candidates(folder_path, patterns)
        if not applicable_images:
            applicable_images = [resolved_file]
        existing_config = find_existing_scale_bar_config_for_image(
            folder_path,
            resolved_file,
            applicable_images,
        )
        return resolved_file, applicable_images, folder_path, existing_config

    if not source_input.is_dir():
        raise RuntimeError(f"Source input must be an image file or a directory: {source_input}")

    resolved_dir = source_input.resolve()
    images = collect_image_candidates(resolved_dir, patterns)
    if not images:
        raise RuntimeError(f"No images found in directory: {source_input}")

    for existing_config in find_existing_scale_bar_configs(resolved_dir):
        existing_image = image_path_from_existing_scale_bar_config(
            existing_config,
            resolved_dir,
            images,
        )
        if existing_image is not None:
            return existing_image, images, resolved_dir, existing_config

    image_index = max(0, as_int(params, "image_index", 0))
    if image_index >= len(images):
        image_index = len(images) - 1
    existing_config = find_existing_scale_bar_config_for_image(
        resolved_dir,
        images[image_index],
        images,
    )
    return images[image_index], images, resolved_dir, existing_config


def parse_point_value(raw: Any, key_name: str) -> tuple[float, float]:
    if isinstance(raw, dict):
        if "x" not in raw or "y" not in raw:
            raise RuntimeError(f"{key_name} dict must contain x and y.")
        return float(raw["x"]), float(raw["y"])

    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return float(raw[0]), float(raw[1])

    raise RuntimeError(f"{key_name} must be an object {{x, y}} or a list [x, y].")


def parse_json_or_value(raw: Any) -> Any:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse JSON value: {text}") from exc
    return raw


def parse_points_from_params(params: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if "point_a" in params and "point_b" in params:
        point_a = parse_point_value(params["point_a"], "point_a")
        point_b = parse_point_value(params["point_b"], "point_b")
        return point_a, point_b

    if "point_a_json" in params and "point_b_json" in params:
        raw_a = parse_json_or_value(params.get("point_a_json"))
        raw_b = parse_json_or_value(params.get("point_b_json"))
        if raw_a is not None and raw_b is not None:
            point_a = parse_point_value(raw_a, "point_a_json")
            point_b = parse_point_value(raw_b, "point_b_json")
            return point_a, point_b

    if "points_json" in params:
        raw_points = parse_json_or_value(params.get("points_json"))
        if isinstance(raw_points, list) and len(raw_points) >= 2:
            point_a = parse_point_value(raw_points[0], "points_json[0]")
            point_b = parse_point_value(raw_points[1], "points_json[1]")
            return point_a, point_b

    if "points" in params:
        raw_points = params.get("points")
        if isinstance(raw_points, list) and len(raw_points) >= 2:
            point_a = parse_point_value(raw_points[0], "points[0]")
            point_b = parse_point_value(raw_points[1], "points[1]")
            return point_a, point_b

    return None


def build_result_payload(
    *,
    source_input: Path,
    image_path: Path,
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    known_length: float,
    length_unit: str,
    params: dict[str, Any],
    selection_mode: str,
    applicable_image_paths: list[Path],
    applies_to_folder: Path | None,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if known_length <= 0:
        raise RuntimeError("known_length must be greater than zero.")

    if image_size is None:
        Image = require_pillow_image()
        with Image.open(image_path) as image:
            image_width, image_height = image.size
    else:
        image_width, image_height = image_size

    dx = float(point_b[0]) - float(point_a[0])
    dy = float(point_b[1]) - float(point_a[1])
    pixel_distance = float(math.hypot(dx, dy))
    if pixel_distance <= 0:
        raise RuntimeError("Selected points must not be identical.")

    pixels_per_unit = pixel_distance / known_length
    units_per_pixel = known_length / pixel_distance
    preview_limit = max(
        1,
        as_int(params, "applies_to_preview_limit", APPLIES_TO_PREVIEW_LIMIT_DEFAULT),
    )
    selected_index = -1
    for index, candidate in enumerate(applicable_image_paths):
        if candidate.resolve() == image_path.resolve():
            selected_index = index
            break

    scope = "single_image"
    if applies_to_folder is not None:
        scope = "all_images_in_folder"

    return {
        "stage_id": "scale_bar_config",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_mode": selection_mode,
        "source_input": str(source_input),
        "selected_image_path": str(image_path),
        "selected_image_name": image_path.name,
        "image_size": {
            "width": int(image_width),
            "height": int(image_height),
        },
        "scale_bar": {
            "point_a": {"x": float(point_a[0]), "y": float(point_a[1])},
            "point_b": {"x": float(point_b[0]), "y": float(point_b[1])},
            "pixel_distance": pixel_distance,
            "known_length": float(known_length),
            "length_unit": str(length_unit),
            "pixels_per_unit": pixels_per_unit,
            "units_per_pixel": units_per_pixel,
        },
        "applies_to": {
            "scope": scope,
            "folder_path": str(applies_to_folder) if applies_to_folder is not None else None,
            "image_count": len(applicable_image_paths),
            "reference_image_index": selected_index,
            "image_glob_used": as_str(params, "image_glob", IMAGE_GLOBS_DEFAULT),
            "image_names_preview": [path.name for path in applicable_image_paths[:preview_limit]],
            "preview_truncated": len(applicable_image_paths) > preview_limit,
        },
        "parameters_used": {
            "image_index": max(0, as_int(params, "image_index", 0)),
            "image_glob": as_str(params, "image_glob", IMAGE_GLOBS_DEFAULT),
            "applies_to_preview_limit": preview_limit,
            "open_results": as_bool(params, "open_results", False),
        },
    }


def direct_output_path_for_image(image_path: Path) -> Path:
    stem = image_path.stem or "scale_bar_config"
    return image_path.parent / f"{stem}.scale_bar_config.result.json"


def direct_output_path_for_payload(payload: dict[str, Any]) -> Path:
    selected_image = str(payload.get("selected_image_path") or "").strip()
    if selected_image:
        image_path = Path(selected_image).expanduser().resolve()
        return direct_output_path_for_image(image_path)

    source_input = str(payload.get("source_input") or "").strip()
    output_dir = Path(source_input).expanduser().resolve().parent if source_input else Path.cwd().resolve()
    stem = Path(source_input).stem if source_input else "scale_bar_config"
    stem = stem or "scale_bar_config"
    return output_dir / f"{stem}.scale_bar_config.result.json"


def write_result_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def point_is_inside_image(point: tuple[float, float], image_size: tuple[int, int]) -> bool:
    return 0 <= point[0] < image_size[0] and 0 <= point[1] < image_size[1]


def parse_scale_bar_preload_payload(
    payload: dict[str, Any],
    image_path: Path,
    image_size: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float], float, str] | None:
    if payload.get("stage_id") not in {None, "scale_bar_config"}:
        return None

    selected_image = str(payload.get("selected_image_path") or "").strip()
    selected_image_name = str(payload.get("selected_image_name") or "").strip()
    if selected_image:
        try:
            selected_path = Path(selected_image).expanduser().resolve()
            if selected_path != image_path.resolve() and selected_image_name != image_path.name:
                return None
        except Exception:
            if selected_image_name != image_path.name:
                return None
    elif selected_image_name and selected_image_name != image_path.name:
        return None

    scale_bar = payload.get("scale_bar")
    if not isinstance(scale_bar, dict):
        return None

    try:
        point_a = parse_point_value(scale_bar.get("point_a"), "scale_bar.point_a")
        point_b = parse_point_value(scale_bar.get("point_b"), "scale_bar.point_b")
        known_length = float(scale_bar.get("known_length"))
    except Exception:
        return None

    if known_length <= 0:
        return None
    if not point_is_inside_image(point_a, image_size) or not point_is_inside_image(point_b, image_size):
        return None

    length_unit = str(scale_bar.get("length_unit") or "mm").strip() or "mm"
    return point_a, point_b, known_length, length_unit


def load_scale_bar_preload(
    image_path: Path,
    image_size: tuple[int, int],
    candidate_paths: list[Path],
) -> tuple[tuple[float, float], tuple[float, float], float, str] | None:
    seen: set[str] = set()
    for path in candidate_paths:
        resolved_path = path.expanduser().resolve()
        key = str(resolved_path).lower()
        if key in seen or not resolved_path.is_file():
            continue
        seen.add(key)

        try:
            payload = load_json_object(resolved_path)
        except Exception:
            continue

        parsed = parse_scale_bar_preload_payload(payload, image_path, image_size)
        if parsed is not None:
            return parsed

    return None


def launch_gui(initial_source_input: Path | None, params: dict[str, Any], existing_result_path: Path | None = None) -> dict[str, Any]:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from PIL import ImageTk
    except Exception as exc:
        raise RuntimeError(
            "GUI mode requires tkinter and Pillow ImageTk support in this environment."
        ) from exc
    Image = require_pillow_image()

    source_input: Path | None = initial_source_input
    if source_input is None:
        chooser = tk.Tk()
        chooser.withdraw()
        chooser.attributes("-topmost", True)
        chooser.update_idletasks()
        chosen_image = filedialog.askopenfilename(
            parent=chooser,
            title="Choose a reference image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        chooser.destroy()
        if not chosen_image:
            raise RuntimeError("No source image selected.")
        source_input = Path(chosen_image).expanduser().resolve()
    else:
        source_input = source_input.expanduser().resolve()

    image_path, applicable_image_paths, applies_to_folder, existing_config_path = resolve_reference_and_applicability(
        source_input,
        params,
    )

    with Image.open(image_path) as source_image:
        image_rgb = source_image.convert("RGB")
        original_size = source_image.size

    max_width = max(400, as_int(params, "preview_max_width", 1400))
    max_height = max(300, as_int(params, "preview_max_height", 900))
    display_size = original_size
    viewport_size = (
        max(1, min(display_size[0], max_width)),
        max(1, min(display_size[1], max_height)),
    )
    preview_image = image_rgb

    navigator_max_width = max(160, as_int(params, "navigator_max_width", 320))
    navigator_max_height = max(120, as_int(params, "navigator_max_height", 240))
    navigator_scale = min(
        1.0,
        navigator_max_width / float(original_size[0]),
        navigator_max_height / float(original_size[1]),
    )
    navigator_size = (
        max(1, int(round(original_size[0] * navigator_scale))),
        max(1, int(round(original_size[1] * navigator_scale))),
    )
    if navigator_size != original_size:
        navigator_image = image_rgb.resize(navigator_size, Image.Resampling.LANCZOS)
    else:
        navigator_image = image_rgb

    preload_candidates: list[Path] = []
    if existing_result_path is not None:
        preload_candidates.append(existing_result_path)
    if existing_config_path is not None:
        preload_candidates.append(existing_config_path)
    preload_candidates.append(direct_output_path_for_image(image_path))
    if applies_to_folder is not None:
        preload_candidates.extend(find_existing_scale_bar_configs(applies_to_folder))
    preloaded_config = load_scale_bar_preload(image_path, original_size, preload_candidates)

    if preloaded_config is not None:
        preloaded_points = (preloaded_config[0], preloaded_config[1])
        known_length_default = preloaded_config[2]
        unit_default = preloaded_config[3]
    else:
        preloaded_points = parse_points_from_params(params)
        known_length_default = as_float(params, "known_length", 1.0)
        if known_length_default <= 0:
            known_length_default = 1.0
        unit_default = as_str(params, "length_unit", "mm").strip() or "mm"

    class ScaleBarWindow:
        def __init__(self) -> None:
            self.root = tk.Tk()
            self.root.title(f"Scale Bar Config {__version__}")
            self.root.geometry("1700x980")
            self.root.after(50, self.open_full_screen)
            self.root.after(100, lambda: bring_tk_window_to_front(self.root))
            self.result: dict[str, Any] | None = None
            self.points: list[tuple[float, float]] = []

            self.known_length_var = tk.StringVar(value=f"{known_length_default}")
            self.unit_var = tk.StringVar(value=unit_default)
            self.distance_var = tk.StringVar(value="Pixel distance: not set")
            self.scale_var = tk.StringVar(value="Scale: not set")

            outer = ttk.Frame(self.root, padding=10)
            outer.pack(fill="both", expand=True)

            controls = ttk.Frame(outer, width=360)
            controls.pack(side="left", fill="y")

            preview = ttk.Frame(outer)
            preview.pack(side="right", fill="both", expand=True, padx=(12, 0))

            ttk.Label(controls, text="Scale Bar Setup", font=("Segoe UI", 12, "bold")).pack(anchor="w")
            ttk.Label(controls, text=f"Image: {image_path.name}", wraplength=340).pack(anchor="w", pady=(8, 2))
            if applies_to_folder is not None:
                ttk.Label(
                    controls,
                    text=(
                        "Applies to all images in folder: "
                        f"{applies_to_folder} (count: {len(applicable_image_paths)})"
                    ),
                    wraplength=340,
                ).pack(anchor="w", pady=(0, 6))
            ttk.Label(
                controls,
                text="Click two points on the 1:1 image. Use the scrollbars or overview box to pan.",
                wraplength=340,
            ).pack(anchor="w", pady=(0, 10))

            ttk.Label(controls, text="Overview").pack(anchor="w", pady=(0, 4))
            self.navigator_photo = ImageTk.PhotoImage(navigator_image)
            self.navigator_canvas = tk.Canvas(
                controls,
                width=navigator_size[0],
                height=navigator_size[1],
                background="#101010",
                highlightthickness=1,
                highlightbackground="#707070",
            )
            self.navigator_canvas.pack(anchor="w", pady=(0, 12))
            self.navigator_canvas.create_image(0, 0, anchor="nw", image=self.navigator_photo)
            self.navigator_canvas.bind("<Button-1>", self.on_navigator_focus)
            self.navigator_canvas.bind("<B1-Motion>", self.on_navigator_focus)

            ttk.Label(controls, text="Known Length").pack(anchor="w")
            ttk.Entry(controls, textvariable=self.known_length_var, width=24).pack(anchor="w", pady=(2, 8))

            ttk.Label(controls, text="Length Unit").pack(anchor="w")
            ttk.Entry(controls, textvariable=self.unit_var, width=24).pack(anchor="w", pady=(2, 12))

            ttk.Label(controls, textvariable=self.distance_var, wraplength=340).pack(anchor="w", pady=(0, 4))
            ttk.Label(controls, textvariable=self.scale_var, wraplength=340).pack(anchor="w", pady=(0, 12))

            button_row = ttk.Frame(controls)
            button_row.pack(anchor="w", pady=(4, 0))
            ttk.Button(button_row, text="Reset Points", command=self.reset_points).pack(side="left")
            ttk.Button(button_row, text="Accept", command=self.accept).pack(side="left", padx=(8, 0))
            ttk.Button(button_row, text="Cancel", command=self.cancel).pack(side="left", padx=(8, 0))

            ttk.Label(preview, text="Image (1:1)").pack(anchor="w")
            self.photo = ImageTk.PhotoImage(preview_image)
            canvas_box = ttk.Frame(preview)
            canvas_box.pack(fill="both", expand=True, pady=(4, 0))
            canvas_box.rowconfigure(0, weight=1)
            canvas_box.columnconfigure(0, weight=1)

            self.canvas = tk.Canvas(
                canvas_box,
                width=viewport_size[0],
                height=viewport_size[1],
                background="#101010",
                highlightthickness=1,
                highlightbackground="#707070",
                scrollregion=(0, 0, display_size[0], display_size[1]),
            )
            self.canvas.grid(row=0, column=0, sticky="nsew")
            self.y_scrollbar = ttk.Scrollbar(canvas_box, orient="vertical", command=self._canvas_yview)
            self.y_scrollbar.grid(row=0, column=1, sticky="ns")
            self.x_scrollbar = ttk.Scrollbar(canvas_box, orient="horizontal", command=self._canvas_xview)
            self.x_scrollbar.grid(row=1, column=0, sticky="ew")
            self.canvas.configure(
                xscrollcommand=self._on_canvas_xscroll,
                yscrollcommand=self._on_canvas_yscroll,
            )
            self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
            self.canvas.bind("<Button-1>", self.on_click)
            self.canvas.bind("<Button-3>", self.reset_points_from_event)
            self.canvas.bind("<Configure>", lambda _event: self._redraw_navigator_viewport())
            self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
            self.canvas.bind("<Shift-MouseWheel>", self.on_shift_mouse_wheel)
            self.canvas.bind("<Button-4>", self.on_mouse_wheel)
            self.canvas.bind("<Button-5>", self.on_mouse_wheel)

            if preloaded_points is not None:
                self.points = [(float(preloaded_points[0][0]), float(preloaded_points[0][1])), (float(preloaded_points[1][0]), float(preloaded_points[1][1]))]

            self.redraw()
            self.root.after(150, self.center_initial_view)

        def open_full_screen(self) -> None:
            try:
                self.root.state("zoomed")
                return
            except Exception:
                pass

            try:
                self.root.attributes("-zoomed", True)
                return
            except Exception:
                pass

            width = self.root.winfo_screenwidth()
            height = self.root.winfo_screenheight()
            self.root.geometry(f"{width}x{height}+0+0")

        @staticmethod
        def _clamp(value: float, lower: float, upper: float) -> float:
            return max(lower, min(upper, value))

        def _to_canvas_xy(self, point: tuple[float, float]) -> tuple[float, float]:
            return point[0], point[1]

        def _to_navigator_xy(self, point: tuple[float, float]) -> tuple[float, float]:
            return point[0] * navigator_scale, point[1] * navigator_scale

        def _on_canvas_xscroll(self, first: str, last: str) -> None:
            self.x_scrollbar.set(first, last)
            self.root.after_idle(self._redraw_navigator_viewport)

        def _on_canvas_yscroll(self, first: str, last: str) -> None:
            self.y_scrollbar.set(first, last)
            self.root.after_idle(self._redraw_navigator_viewport)

        def _canvas_xview(self, *args: Any) -> None:
            self.canvas.xview(*args)
            self.root.after_idle(self._redraw_navigator_viewport)

        def _canvas_yview(self, *args: Any) -> None:
            self.canvas.yview(*args)
            self.root.after_idle(self._redraw_navigator_viewport)

        def _redraw_navigator_marks(self) -> None:
            self.navigator_canvas.delete("navigator_marks")
            if len(self.points) == 2:
                p0x, p0y = self._to_navigator_xy(self.points[0])
                p1x, p1y = self._to_navigator_xy(self.points[1])
                self.navigator_canvas.create_line(
                    p0x,
                    p0y,
                    p1x,
                    p1y,
                    fill="#ffd166",
                    width=2,
                    tags="navigator_marks",
                )

            for index, point in enumerate(self.points):
                cx, cy = self._to_navigator_xy(point)
                radius = 3
                color = "#50e3c2" if index == 0 else "#ff8c42"
                self.navigator_canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    outline=color,
                    width=2,
                    tags="navigator_marks",
                )

        def _redraw_navigator_viewport(self) -> None:
            if not hasattr(self, "navigator_canvas") or not hasattr(self, "canvas"):
                return

            self.navigator_canvas.delete("navigator_viewport")
            canvas_w = max(self.canvas.winfo_width(), 1)
            canvas_h = max(self.canvas.winfo_height(), 1)
            left = self._clamp(float(self.canvas.canvasx(0)), 0.0, float(display_size[0]))
            top = self._clamp(float(self.canvas.canvasy(0)), 0.0, float(display_size[1]))
            right = self._clamp(float(self.canvas.canvasx(canvas_w)), 0.0, float(display_size[0]))
            bottom = self._clamp(float(self.canvas.canvasy(canvas_h)), 0.0, float(display_size[1]))

            x0 = self._clamp(left * navigator_scale, 0.0, float(navigator_size[0] - 1))
            y0 = self._clamp(top * navigator_scale, 0.0, float(navigator_size[1] - 1))
            x1 = self._clamp(right * navigator_scale, 0.0, float(navigator_size[0] - 1))
            y1 = self._clamp(bottom * navigator_scale, 0.0, float(navigator_size[1] - 1))
            rect_x1 = self._clamp(max(x0 + 1.0, x1), 0.0, float(navigator_size[0] - 1))
            rect_y1 = self._clamp(max(y0 + 1.0, y1), 0.0, float(navigator_size[1] - 1))
            self.navigator_canvas.create_rectangle(
                x0,
                y0,
                rect_x1,
                rect_y1,
                outline="#ff0000",
                width=2,
                tags="navigator_viewport",
            )

        def _center_canvas_on_image_point(self, x: float, y: float) -> None:
            canvas_w = max(self.canvas.winfo_width(), 1)
            canvas_h = max(self.canvas.winfo_height(), 1)
            max_left = max(0.0, float(display_size[0] - canvas_w))
            max_top = max(0.0, float(display_size[1] - canvas_h))
            left = self._clamp(x - canvas_w / 2.0, 0.0, max_left)
            top = self._clamp(y - canvas_h / 2.0, 0.0, max_top)

            if display_size[0] > canvas_w:
                self.canvas.xview_moveto(left / float(display_size[0]))
            else:
                self.canvas.xview_moveto(0.0)
            if display_size[1] > canvas_h:
                self.canvas.yview_moveto(top / float(display_size[1]))
            else:
                self.canvas.yview_moveto(0.0)
            self._redraw_navigator_viewport()

        def center_on_points(self) -> None:
            if not self.points:
                return
            x = sum(point[0] for point in self.points) / float(len(self.points))
            y = sum(point[1] for point in self.points) / float(len(self.points))
            self._center_canvas_on_image_point(x, y)

        def center_on_image_middle(self) -> None:
            self._center_canvas_on_image_point(display_size[0] / 2.0, display_size[1] / 2.0)

        def center_initial_view(self) -> None:
            if self.points:
                self.center_on_points()
            else:
                self.center_on_image_middle()

        def on_navigator_focus(self, event: Any) -> str:
            nav_x = self._clamp(float(event.x), 0.0, float(navigator_size[0]))
            nav_y = self._clamp(float(event.y), 0.0, float(navigator_size[1]))
            self._center_canvas_on_image_point(nav_x / navigator_scale, nav_y / navigator_scale)
            return "break"

        def on_mouse_wheel(self, event: Any) -> str:
            units = self._wheel_units(event)
            if units:
                self.canvas.yview_scroll(units, "units")
                self._redraw_navigator_viewport()
            return "break"

        def on_shift_mouse_wheel(self, event: Any) -> str:
            units = self._wheel_units(event)
            if units:
                self.canvas.xview_scroll(units, "units")
                self._redraw_navigator_viewport()
            return "break"

        @staticmethod
        def _wheel_units(event: Any) -> int:
            event_num = getattr(event, "num", None)
            if event_num == 4:
                return -3
            if event_num == 5:
                return 3

            delta = int(getattr(event, "delta", 0))
            if delta == 0:
                return 0
            units = -int(delta / 120)
            if units == 0:
                units = -1 if delta > 0 else 1
            return units

        def redraw(self) -> None:
            self.canvas.delete("scale_overlay")

            for index, point in enumerate(self.points):
                cx, cy = self._to_canvas_xy(point)
                radius = 5
                color = "#50e3c2" if index == 0 else "#ff8c42"
                self.canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    outline=color,
                    width=2,
                    tags="scale_overlay",
                )

            if len(self.points) == 2:
                p0x, p0y = self._to_canvas_xy(self.points[0])
                p1x, p1y = self._to_canvas_xy(self.points[1])
                self.canvas.create_line(
                    p0x,
                    p0y,
                    p1x,
                    p1y,
                    fill="#ffd166",
                    width=2,
                    tags="scale_overlay",
                )
                pixel_distance = float(math.hypot(
                    self.points[1][0] - self.points[0][0],
                    self.points[1][1] - self.points[0][1],
                ))
                self.distance_var.set(f"Pixel distance: {pixel_distance:.4f}px")
                known = self._known_length_or_none()
                if known is not None:
                    unit = self.unit_var.get().strip() or "units"
                    self.scale_var.set(
                        f"Scale: {pixel_distance / known:.6f} px/{unit} ({known / pixel_distance:.6f} {unit}/px)"
                    )
                else:
                    self.scale_var.set("Scale: enter a known length greater than zero.")
            else:
                self.distance_var.set("Pixel distance: not set")
                self.scale_var.set("Scale: not set")

            self._redraw_navigator_marks()
            self._redraw_navigator_viewport()

        def _known_length_or_none(self) -> float | None:
            try:
                value = float(self.known_length_var.get())
            except ValueError:
                return None
            if value <= 0:
                return None
            return value

        def on_click(self, event: Any) -> None:
            x = float(self.canvas.canvasx(event.x))
            y = float(self.canvas.canvasy(event.y))
            if x < 0 or y < 0 or x >= display_size[0] or y >= display_size[1]:
                return

            point = (x, y)
            if len(self.points) >= 2:
                self.points = [point]
            else:
                self.points.append(point)
            self.redraw()

        def reset_points(self) -> None:
            self.points = []
            self.redraw()

        def reset_points_from_event(self, _event: Any) -> str:
            self.reset_points()
            return "break"

        def accept(self) -> None:
            if len(self.points) != 2:
                messagebox.showerror("Scale Bar Config", "Select exactly two points before accepting.")
                return

            known_length = self._known_length_or_none()
            if known_length is None:
                messagebox.showerror("Scale Bar Config", "Known length must be a positive number.")
                return

            length_unit = self.unit_var.get().strip() or "units"
            self.result = build_result_payload(
                source_input=source_input,
                image_path=image_path,
                point_a=self.points[0],
                point_b=self.points[1],
                known_length=known_length,
                length_unit=length_unit,
                params=params,
                selection_mode="gui",
                applicable_image_paths=applicable_image_paths,
                applies_to_folder=applies_to_folder,
                image_size=original_size,
            )
            self.root.destroy()

        def cancel(self) -> None:
            self.root.destroy()

        def run(self) -> dict[str, Any]:
            self.root.mainloop()
            if self.result is None:
                raise RuntimeError("Scale bar setup was cancelled before completion.")
            return self.result

    window = ScaleBarWindow()
    return window.run()


def run_stage_for_runner(source_input: Path, params: dict[str, Any], output_result_json: Path | None = None) -> dict[str, Any]:
    run_mode = resolve_run_mode()
    open_results = as_bool(params, "open_results", False)
    if run_mode == "visual" or open_results:
        payload = launch_gui(source_input, params, output_result_json)
        payload["action"] = "open_results"
        payload["run_mode"] = run_mode
        payload["open_results"] = open_results
        return payload

    points = parse_points_from_params(params)
    if points is None:
        raise RuntimeError(
            "Headless mode needs scale points in params. Provide one of: "
            "point_a + point_b, point_a_json + point_b_json, points, or points_json."
        )

    known_length = as_float(params, "known_length", 0.0)
    if known_length <= 0:
        raise RuntimeError("Headless mode requires params.known_length > 0.")
    length_unit = as_str(params, "length_unit", "mm").strip() or "mm"

    image_path, applicable_image_paths, applies_to_folder, _existing_config_path = resolve_reference_and_applicability(
        source_input,
        params,
    )
    payload = build_result_payload(
        source_input=source_input,
        image_path=image_path,
        point_a=points[0],
        point_b=points[1],
        known_length=known_length,
        length_unit=length_unit,
        params=params,
        selection_mode="params",
        applicable_image_paths=applicable_image_paths,
        applies_to_folder=applies_to_folder,
    )
    payload["action"] = "process"
    payload["run_mode"] = run_mode
    payload["open_results"] = open_results
    return payload


def main() -> int:
    args = parse_args()

    has_source = has_text(args.source_input)
    has_output = has_text(args.output_result_json)
    has_params = has_text(args.params_json)

    full_runner_mode = has_source and has_output and has_params

    if args.gui:
        gui_source_raw = args.gui_source_input or args.source_input or ""
        gui_source = Path(gui_source_raw).expanduser().resolve() if has_text(gui_source_raw) else None
        gui_params_raw = args.gui_params_json or args.params_json or ""
        gui_params = load_json_object_optional(gui_params_raw)
        explicit_output_result_json = (
            Path(str(args.output_result_json)).expanduser().resolve()
            if has_output
            else None
        )
        payload = launch_gui(gui_source, gui_params, explicit_output_result_json)
        output_result_json = explicit_output_result_json or direct_output_path_for_payload(payload)
        payload["output_result_json"] = str(output_result_json)
        write_result_json(output_result_json, payload)
        print(f"Wrote result JSON: {output_result_json}", flush=True)
        print(json.dumps(payload, indent=2), flush=True)
        return 0

    if full_runner_mode:
        source_input = Path(str(args.source_input)).expanduser().resolve()
        if not source_input.exists():
            raise RuntimeError(f"Source input does not exist: {source_input}")

        output_result_json = Path(str(args.output_result_json)).expanduser().resolve()
        params_path = Path(str(args.params_json)).expanduser().resolve()
        params = load_json_object(params_path)

        payload = run_stage_for_runner(source_input, params, output_result_json)
        payload.setdefault("source_exists", source_input.exists())

        write_result_json(output_result_json, payload)
        print(f"Wrote result JSON: {output_result_json}", flush=True)
        return 0

    # Direct/manual mode:
    # - no runner args
    # - source-only
    # - params-only (used as GUI preload)
    if has_output:
        raise RuntimeError(
            "Partial runner args detected. If --output-result-json is provided, also provide "
            "--source-input and --params-json."
        )

    gui_source_raw = args.gui_source_input or args.source_input or ""
    gui_source = Path(gui_source_raw).expanduser().resolve() if has_text(gui_source_raw) else None
    gui_params_raw = args.gui_params_json or args.params_json or ""
    gui_params = load_json_object_optional(gui_params_raw)
    payload = launch_gui(gui_source, gui_params)
    output_result_json = direct_output_path_for_payload(payload)
    payload["output_result_json"] = str(output_result_json)
    write_result_json(output_result_json, payload)
    print(f"Wrote result JSON: {output_result_json}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
