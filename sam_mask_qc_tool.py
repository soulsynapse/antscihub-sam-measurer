from __future__ import annotations

import argparse
import ast
import csv
import json
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageTk

import sam_mask_engine as mask_engine


DEFAULT_OUTPUT_CSV_NAME = "sam_mask_measurements.csv"
ANNOTATION_FILE_SUFFIX = ".sam_clicks.npz"
ANNOTATION_METADATA_FILE_SUFFIX = ".sam_clicks.json"
SCALE_BAR_CONFIG_SUFFIX = ".scale_bar_config.result.json"
BACKGROUND_DIM_FACTOR = 0.25
BBOX_BORDER_WIDTH = 1
ZOOM_STEP_FACTOR = 1.12

# Keep these labels/colors matched with sam_hover_mask_gui.py without importing
# the full model-loading GUI.
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


@dataclass(frozen=True)
class MaskEntry:
    row: dict[str, Any]
    field_order: list[str]
    image_name: str
    mask_number: int
    mask_index: int
    image_path: Path
    metadata_path: Path
    annotation_path: Path


@dataclass(frozen=True)
class ScaleCalibration:
    path: Path | None
    units_per_pixel: float
    length_unit: str
    origin_image_path: Path | None = None
    origin_image_name: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse SAM mask measurement outputs with 1:1 bbox previews.",
    )
    parser.add_argument(
        "source_folder",
        nargs="?",
        help="Folder containing sam_mask_measurements.csv and SAM annotation files.",
    )
    parser.add_argument(
        "--source-folder",
        dest="source_folder_option",
        help="Folder containing sam_mask_measurements.csv and SAM annotation files.",
    )
    parser.add_argument(
        "--output-file",
        "--output-csv",
        dest="output_file",
        help=(
            "Measurement CSV to read. Defaults to "
            f"{DEFAULT_OUTPUT_CSV_NAME} inside the source folder."
        ),
    )
    return parser.parse_args()


def bring_tk_window_to_front(window: Any) -> None:
    try:
        window.deiconify()
        window.lift()
        window.attributes("-topmost", True)
        window.focus_force()
        window.after(600, lambda: window.attributes("-topmost", False))
    except Exception:
        pass


def maximize_tk_window(window: Any) -> None:
    try:
        window.state("zoomed")
        return
    except Exception:
        pass

    try:
        window.attributes("-zoomed", True)
        return
    except Exception:
        pass

    try:
        width = window.winfo_screenwidth()
        height = window.winfo_screenheight()
        window.geometry(f"{width}x{height}+0+0")
    except Exception:
        pass


def parse_int_like(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return default


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{\"'":
        return value
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return value


def parse_bbox(value: Any) -> list[int] | None:
    raw = parse_jsonish(value)
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    bbox: list[int] = []
    for item in raw:
        parsed = parse_int_like(item)
        if parsed is None:
            return None
        bbox.append(int(parsed))
    return bbox


def format_value(value: Any) -> str:
    value = parse_jsonish(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def resolve_path(raw_path: Any, folder: Path) -> Path | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate
    return (folder / candidate).resolve()


def metadata_path_for_image(image_path: Path) -> Path:
    return Path(str(image_path) + ANNOTATION_METADATA_FILE_SUFFIX)


def annotation_path_for_image(image_path: Path) -> Path:
    return Path(str(image_path) + ANNOTATION_FILE_SUFFIX)


def annotation_path_for_metadata(metadata_path: Path) -> Path:
    text = str(metadata_path)
    if text.endswith(ANNOTATION_METADATA_FILE_SUFFIX):
        return Path(text[: -len(ANNOTATION_METADATA_FILE_SUFFIX)] + ANNOTATION_FILE_SUFFIX)
    return Path(text + ANNOTATION_FILE_SUFFIX)


def image_name_from_metadata_path(metadata_path: Path) -> str:
    name = metadata_path.name
    if name.endswith(ANNOTATION_METADATA_FILE_SUFFIX):
        return name[: -len(ANNOTATION_METADATA_FILE_SUFFIX)]
    return name


def choose_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_entry_from_csv_row(
    folder: Path,
    row: dict[str, Any],
    field_order: list[str],
) -> MaskEntry:
    image_name = str(row.get("image_name") or row.get("annotation_image_name") or "").strip()
    metadata_candidate = resolve_path(row.get("annotation_metadata_path"), folder)
    image_candidate = resolve_path(row.get("annotation_image_path"), folder)
    if image_candidate is None and image_name:
        image_candidate = (folder / image_name).resolve()
    if metadata_candidate is None and image_candidate is not None:
        metadata_candidate = metadata_path_for_image(image_candidate)
    if metadata_candidate is not None and image_candidate is None:
        image_candidate = (metadata_candidate.parent / image_name_from_metadata_path(metadata_candidate)).resolve()
    if image_candidate is None:
        raise RuntimeError(f"Could not resolve image path for row: {row!r}")
    if metadata_candidate is None:
        metadata_candidate = metadata_path_for_image(image_candidate)

    metadata_path = choose_existing_path(
        [
            metadata_candidate,
            metadata_path_for_image(image_candidate),
            image_candidate.parent / f"{image_candidate.name}{ANNOTATION_METADATA_FILE_SUFFIX}",
        ]
    )
    annotation_path = choose_existing_path(
        [
            annotation_path_for_metadata(metadata_path),
            annotation_path_for_image(image_candidate),
            image_candidate.parent / f"{image_candidate.name}{ANNOTATION_FILE_SUFFIX}",
        ]
    )
    if not image_name:
        image_name = image_candidate.name

    mask_index = parse_int_like(row.get("record_mask_index"))
    click_number = parse_int_like(row.get("click_number"))
    if mask_index is None:
        mask_index = max(0, int(click_number) - 1) if click_number is not None else 0
    mask_number = int(click_number) if click_number is not None else int(mask_index) + 1
    return MaskEntry(
        row=dict(row),
        field_order=list(field_order),
        image_name=image_name,
        mask_number=mask_number,
        mask_index=int(mask_index),
        image_path=image_candidate,
        metadata_path=metadata_path,
        annotation_path=annotation_path,
    )


def load_csv_entries(folder: Path, csv_path: Path) -> list[MaskEntry]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        field_order = list(reader.fieldnames or [])
        entries = [
            build_entry_from_csv_row(folder, dict(row), field_order)
            for row in reader
        ]
    if not entries:
        raise RuntimeError(f"No mask rows found in output file: {csv_path}")
    return entries


def stringify_row_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def scale_calibration_from_row(row: dict[str, Any], folder: Path) -> ScaleCalibration | None:
    units_per_pixel = None
    for key in (
        "ratio_units_per_pixel",
        "scale_config_scale_bar_units_per_pixel",
        "scale_bar_units_per_pixel",
    ):
        units_per_pixel = mask_engine.parse_float_like(row.get(key))
        if units_per_pixel is not None:
            break
    if units_per_pixel is None or units_per_pixel <= 0:
        return None

    raw_unit = (
        row.get("scale_config_scale_bar_length_unit")
        or row.get("scale_bar_length_unit")
        or row.get("computed_area_unit", "").replace("^2", "")
        or "units"
    )
    length_unit = str(raw_unit).strip() or "units"
    scale_path = resolve_path(row.get("scale_bar_config_path"), folder)
    origin_image_path = resolve_path(
        row.get("scale_config_selected_image_path") or row.get("selected_image_path"),
        folder,
    )
    raw_origin_name = (
        row.get("scale_config_selected_image_name")
        or row.get("selected_image_name")
        or (origin_image_path.name if origin_image_path is not None else "")
    )
    origin_image_name = str(raw_origin_name).strip()
    return ScaleCalibration(
        scale_path,
        float(units_per_pixel),
        length_unit,
        origin_image_path,
        origin_image_name,
    )


def read_scale_calibration(path: Path) -> ScaleCalibration | None:
    try:
        payload = load_json_object(path)
    except Exception:
        return None
    scale_bar = payload.get("scale_bar")
    if not isinstance(scale_bar, dict):
        return None
    units_per_pixel = mask_engine.parse_float_like(scale_bar.get("units_per_pixel"))
    if units_per_pixel is None or units_per_pixel <= 0:
        return None
    length_unit = str(scale_bar.get("length_unit") or "units").strip() or "units"
    origin_image_path = resolve_path(payload.get("selected_image_path"), path.parent)
    raw_origin_name = (
        payload.get("selected_image_name")
        or (origin_image_path.name if origin_image_path is not None else "")
    )
    origin_image_name = str(raw_origin_name).strip()
    return ScaleCalibration(
        path,
        float(units_per_pixel),
        length_unit,
        origin_image_path,
        origin_image_name,
    )


def candidate_scale_bar_configs(folder: Path, entry: MaskEntry) -> list[Path]:
    candidates: list[Path] = []
    for raw_path in (
        entry.row.get("scale_bar_config_path"),
        entry.row.get("scale_config_output_result_json"),
    ):
        resolved = resolve_path(raw_path, folder)
        if resolved is not None:
            candidates.append(resolved)

    image_stem_candidate = entry.image_path.with_name(f"{entry.image_path.stem}{SCALE_BAR_CONFIG_SUFFIX}")
    candidates.append(image_stem_candidate)
    candidates.extend(sorted(folder.glob(f"*{SCALE_BAR_CONFIG_SUFFIX}")))
    if entry.image_path.parent != folder:
        candidates.extend(sorted(entry.image_path.parent.glob(f"*{SCALE_BAR_CONFIG_SUFFIX}")))

    unique: dict[str, Path] = {}
    for candidate in candidates:
        unique[str(candidate.resolve())] = candidate.resolve()
    return list(unique.values())


def resolve_scale_calibration(folder: Path, entry: MaskEntry) -> ScaleCalibration | None:
    row_scale = scale_calibration_from_row(entry.row, folder)
    if row_scale is not None:
        if row_scale.origin_image_path is None and not row_scale.origin_image_name:
            if row_scale.path is not None and row_scale.path.exists():
                file_scale = read_scale_calibration(row_scale.path)
                if file_scale is not None:
                    return ScaleCalibration(
                        row_scale.path,
                        row_scale.units_per_pixel,
                        row_scale.length_unit,
                        file_scale.origin_image_path,
                        file_scale.origin_image_name,
                    )
        return row_scale

    existing = [path for path in candidate_scale_bar_configs(folder, entry) if path.exists()]
    existing.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in existing:
        calibration = read_scale_calibration(path)
        if calibration is not None:
            return calibration
    return None


def scale_origin_differs_from_entry(scale: ScaleCalibration | None, entry: MaskEntry) -> bool:
    if scale is None:
        return False
    if scale.origin_image_name:
        return scale.origin_image_name != entry.image_path.name
    if scale.origin_image_path is not None:
        if scale.origin_image_path.name == entry.image_path.name:
            return False
        try:
            return scale.origin_image_path.resolve() != entry.image_path.resolve()
        except Exception:
            return True
    return False


def load_annotation_metadata_entries(folder: Path) -> list[MaskEntry]:
    entries: list[MaskEntry] = []
    for metadata_path in sorted(folder.glob(f"*{ANNOTATION_METADATA_FILE_SUFFIX}")):
        payload = load_json_object(metadata_path)
        image_name = str(payload.get("image_name") or image_name_from_metadata_path(metadata_path))
        image_path = resolve_path(payload.get("image_path"), folder)
        if image_path is None or not image_path.exists():
            image_path = (metadata_path.parent / image_name).resolve()
        annotation_path = annotation_path_for_metadata(metadata_path)
        records = payload.get("records", [])
        if not isinstance(records, list):
            continue
        annotation_fields = [
            key for key in payload.keys()
            if key not in {"records", "ignore_records"}
        ]
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            mask_index = parse_int_like(record.get("mask_index"), index)
            if mask_index is None:
                mask_index = index
            mask_number = int(mask_index) + 1
            row: dict[str, Any] = {
                "image_name": image_name,
                "click_number": str(mask_number),
                "annotation_metadata_path": str(metadata_path),
                "annotation_image_path": str(image_path),
            }
            for key in annotation_fields:
                row[f"annotation_{key}"] = stringify_row_value(payload.get(key))
            for key, value in record.items():
                row[f"record_{key}"] = stringify_row_value(value)
            field_order = list(row.keys())
            entries.append(
                MaskEntry(
                    row=row,
                    field_order=field_order,
                    image_name=image_name,
                    mask_number=mask_number,
                    mask_index=int(mask_index),
                    image_path=image_path,
                    metadata_path=metadata_path,
                    annotation_path=annotation_path,
                )
            )
    if not entries:
        raise RuntimeError(
            f"No {DEFAULT_OUTPUT_CSV_NAME} or *{ANNOTATION_METADATA_FILE_SUFFIX} rows found in {folder}"
        )
    return entries


def load_mask_entries(folder: Path, output_file: Path | None = None) -> tuple[list[MaskEntry], Path | None]:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise RuntimeError(f"Source folder does not exist: {folder}")
    csv_path = output_file.expanduser().resolve() if output_file is not None else folder / DEFAULT_OUTPUT_CSV_NAME
    if csv_path.exists():
        return load_csv_entries(folder, csv_path), csv_path
    return load_annotation_metadata_entries(folder), None


def clamp_bbox(bbox: list[int], width: int, height: int) -> list[int] | None:
    if width <= 0 or height <= 0:
        return None
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(width - 1, int(x0)))
    y0 = max(0, min(height - 1, int(y0)))
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def color_for_name(raw_name: str) -> tuple[int, int, int]:
    return OUTLINE_COLOR_MAP.get(raw_name, OUTLINE_COLOR_MAP[DEFAULT_OUTLINE_COLOR_NAME])


def build_preview_image(
    image: Image.Image,
    binary_mask: np.ndarray,
    bbox_xyxy: list[int],
    outline_color: tuple[int, int, int],
) -> Image.Image:
    return build_viewport_image(
        image,
        binary_mask,
        bbox_xyxy,
        outline_color,
        zoom_factor=1.0,
        canvas_view_xy=(float(bbox_xyxy[0]), float(bbox_xyxy[1])),
        canvas_size=(bbox_xyxy[2] - bbox_xyxy[0] + 1, bbox_xyxy[3] - bbox_xyxy[1] + 1),
    )[0]


def build_viewport_image(
    image: Image.Image,
    binary_mask: np.ndarray,
    bbox_xyxy: list[int],
    outline_color: tuple[int, int, int],
    zoom_factor: float,
    canvas_view_xy: tuple[float, float],
    canvas_size: tuple[int, int],
    show_mask: bool = True,
    show_bbox: bool = True,
    caliper_result: mask_engine.MaskCaliperResult | None = None,
    hull_result: mask_engine.MaskHullResult | None = None,
    ellipse_result: mask_engine.MaskEllipseResult | None = None,
) -> tuple[Image.Image, tuple[float, float]]:
    width, height = image.size
    zoom = max(1e-6, float(zoom_factor))
    display_width = max(1, int(round(width * zoom)))
    display_height = max(1, int(round(height * zoom)))
    canvas_width = max(1, int(canvas_size[0]))
    canvas_height = max(1, int(canvas_size[1]))
    view_left = max(0.0, min(float(canvas_view_xy[0]), float(max(0, display_width - 1))))
    view_top = max(0.0, min(float(canvas_view_xy[1]), float(max(0, display_height - 1))))
    view_right = min(float(display_width), view_left + float(canvas_width))
    view_bottom = min(float(display_height), view_top + float(canvas_height))

    src_left = max(0, min(width - 1, int(np.floor(view_left / zoom))))
    src_top = max(0, min(height - 1, int(np.floor(view_top / zoom))))
    src_right = max(src_left + 1, min(width, int(np.ceil(view_right / zoom))))
    src_bottom = max(src_top + 1, min(height, int(np.ceil(view_bottom / zoom))))
    origin_x = float(src_left) * zoom
    origin_y = float(src_top) * zoom

    crop = image.crop((src_left, src_top, src_right, src_bottom)).convert("RGB")
    scaled_size = (
        max(1, int(round(crop.width * zoom))),
        max(1, int(round(crop.height * zoom))),
    )
    if scaled_size != crop.size:
        crop = crop.resize(scaled_size, Image.Resampling.NEAREST)

    mask_crop = Image.fromarray(
        np.asarray(binary_mask[src_top:src_bottom, src_left:src_right], dtype=np.uint8) * 255,
        mode="L",
    )
    if scaled_size != mask_crop.size:
        mask_crop = mask_crop.resize(scaled_size, Image.Resampling.NEAREST)

    frame = np.asarray(crop, dtype=np.uint8).copy()
    mask_crop_arr = np.asarray(mask_crop) > 0
    x0, y0, x1, y1 = bbox_xyxy
    local_x0 = int(round(float(x0) * zoom - origin_x))
    local_y0 = int(round(float(y0) * zoom - origin_y))
    local_x1 = int(round(float(x1 + 1) * zoom - origin_x)) - 1
    local_y1 = int(round(float(y1 + 1) * zoom - origin_y)) - 1
    frame_h, frame_w = frame.shape[:2]
    if show_mask and frame.ndim == 3 and mask_crop_arr.shape == frame.shape[:2] and bool(np.any(mask_crop_arr)):
        bbox_visible = np.zeros(mask_crop_arr.shape, dtype=bool)
        if local_x1 >= 0 and local_y1 >= 0 and local_x0 < frame_w and local_y0 < frame_h:
            rx0 = max(0, local_x0)
            ry0 = max(0, local_y0)
            rx1 = min(frame_w - 1, local_x1)
            ry1 = min(frame_h - 1, local_y1)
            bbox_visible[ry0 : ry1 + 1, rx0 : rx1 + 1] = True
        outside = (~mask_crop_arr) & bbox_visible
        frame[outside] = (
            frame[outside].astype(np.float32) * float(BACKGROUND_DIM_FACTOR)
        ).astype(np.uint8)

    border = max(1, int(BBOX_BORDER_WIDTH))
    color = np.asarray(outline_color, dtype=np.uint8)
    if show_bbox and local_x1 >= 0 and local_y1 >= 0 and local_x0 < frame_w and local_y0 < frame_h:
        rx0 = max(0, local_x0)
        ry0 = max(0, local_y0)
        rx1 = min(frame_w - 1, local_x1)
        ry1 = min(frame_h - 1, local_y1)
        frame[ry0 : min(frame_h, ry0 + border), rx0 : rx1 + 1, :] = color
        frame[max(0, ry1 - border + 1) : ry1 + 1, rx0 : rx1 + 1, :] = color
        frame[ry0 : ry1 + 1, rx0 : min(frame_w, rx0 + border), :] = color
        frame[ry0 : ry1 + 1, max(0, rx1 - border + 1) : rx1 + 1, :] = color
    result = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(result)
    draw_color = tuple(int(channel) for channel in outline_color)
    line_width = max(1, int(BBOX_BORDER_WIDTH))
    if hull_result is not None and len(hull_result.points_xy) >= 2:
        hull_points = [
            (float(x) * zoom - origin_x, float(y) * zoom - origin_y)
            for x, y in hull_result.points_xy
        ]
        draw.line(hull_points + [hull_points[0]], fill=draw_color, width=line_width)
    if ellipse_result is not None:
        theta = np.linspace(0.0, 2.0 * np.pi, 97)
        angle = np.deg2rad(float(ellipse_result.angle_degrees))
        cos_angle = float(np.cos(angle))
        sin_angle = float(np.sin(angle))
        a = float(ellipse_result.major_axis_length_px) * zoom / 2.0
        b = float(ellipse_result.minor_axis_length_px) * zoom / 2.0
        cx = float(ellipse_result.center_xy[0]) * zoom - origin_x
        cy = float(ellipse_result.center_xy[1]) * zoom - origin_y
        ellipse_points = []
        for value in theta:
            local_x = a * float(np.cos(value))
            local_y = b * float(np.sin(value))
            ellipse_points.append(
                (
                    cx + local_x * cos_angle - local_y * sin_angle,
                    cy + local_x * sin_angle + local_y * cos_angle,
                )
            )
        draw.line(ellipse_points, fill=draw_color, width=line_width)
    if caliper_result is not None:
        ax = (float(caliper_result.point_a_xy[0]) + 0.5) * zoom - origin_x
        ay = (float(caliper_result.point_a_xy[1]) + 0.5) * zoom - origin_y
        bx = (float(caliper_result.point_b_xy[0]) + 0.5) * zoom - origin_x
        by = (float(caliper_result.point_b_xy[1]) + 0.5) * zoom - origin_y
        draw.line(
            (ax, ay, bx, by),
            fill=draw_color,
            width=line_width,
        )
    return result, (origin_x, origin_y)


class SamMaskQcApp:
    def __init__(self, root: tk.Tk, folder: Path, entries: list[MaskEntry], source_file: Path | None) -> None:
        self.root = root
        self.folder = folder
        self.entries = entries
        self.source_file = source_file
        self.index = 0
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.image_cache: dict[Path, Image.Image] = {}
        self.annotation_cache: dict[tuple[Path, Path, int, int], mask_engine.AnnotationLoadResult] = {}
        self.zoom_factor = 1.0
        self._zoom_min = 0.2
        self._zoom_max = 8.0
        self._middle_pan_last_canvas_xy: tuple[float, float] | None = None
        self.current_image: Image.Image | None = None
        self.current_binary_mask: np.ndarray | None = None
        self.current_bbox: list[int] | None = None
        self.current_record_metadata: dict[str, Any] = {}
        self.current_entry: MaskEntry | None = None
        self.current_caliper: mask_engine.MaskCaliperResult | None = None
        self.current_hull: mask_engine.MaskHullResult | None = None
        self.current_ellipse: mask_engine.MaskEllipseResult | None = None
        self.raw_view_hold = False
        self._table_value_labels: list[ttk.Label] = []

        self.scale_warning_var = tk.StringVar(value="")
        self.outline_color_var = tk.StringVar(value=DEFAULT_OUTLINE_COLOR_NAME)
        self.show_bbox_var = tk.BooleanVar(value=True)
        self.show_caliper_var = tk.BooleanVar(value=False)
        self.show_hull_var = tk.BooleanVar(value=False)
        self.show_ellipse_var = tk.BooleanVar(value=False)
        self.position_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self._build_ui()
        self._render_current()

    def _build_ui(self) -> None:
        self.root.title("SAM Mask QC")
        self.root.geometry("1200x800")
        self.root.minsize(760, 480)

        pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        preview_frame = ttk.Frame(pane)
        stats_frame = ttk.Frame(pane, width=420)
        pane.add(preview_frame, weight=3)
        pane.add(stats_frame, weight=1)

        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(preview_frame, bg="#111111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self._canvas_yview)
        x_scroll = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self._canvas_xview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

        sidebar_outer = ttk.Frame(stats_frame, padding=(8, 8, 8, 8))
        sidebar_outer.pack(fill=tk.BOTH, expand=True)
        sidebar_outer.rowconfigure(0, weight=1)
        sidebar_outer.columnconfigure(0, weight=1)
        self.sidebar_canvas = tk.Canvas(sidebar_outer, highlightthickness=0)
        self.sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        sidebar_y = ttk.Scrollbar(sidebar_outer, orient=tk.VERTICAL, command=self.sidebar_canvas.yview)
        sidebar_y.grid(row=0, column=1, sticky="ns")
        self.sidebar_canvas.configure(yscrollcommand=sidebar_y.set)
        self.sidebar_content = ttk.Frame(self.sidebar_canvas)
        self.sidebar_window = self.sidebar_canvas.create_window(
            0,
            0,
            anchor="nw",
            window=self.sidebar_content,
        )
        self.sidebar_content.bind("<Configure>", self._on_sidebar_content_configure)
        self.sidebar_canvas.bind("<Configure>", self._on_sidebar_canvas_configure)
        self._bind_sidebar_mousewheel(self.sidebar_canvas)
        self._bind_sidebar_mousewheel(self.sidebar_content)

        controls = ttk.Frame(self.sidebar_content, padding=(0, 0, 0, 8))
        controls.pack(fill=tk.X)
        self._bind_sidebar_mousewheel(controls)
        ttk.Button(controls, text="Open folder", command=self._open_folder).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(controls, text="<", width=4, command=lambda: self._navigate(-1)).pack(
            side=tk.LEFT
        )
        ttk.Label(controls, textvariable=self.position_var, anchor="center").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=8
        )
        ttk.Button(controls, text=">", width=4, command=lambda: self._navigate(1)).pack(
            side=tk.LEFT
        )

        color_row = ttk.Frame(self.sidebar_content, padding=(0, 0, 0, 8))
        color_row.pack(fill=tk.X)
        self._bind_sidebar_mousewheel(color_row)
        ttk.Style(self.root).configure(
            "ScaleWarning.TLabel",
            foreground="#c00000",
            font=("Segoe UI", 10, "bold"),
        )
        ttk.Label(color_row, text="Outline:").pack(side=tk.LEFT, padx=(0, 4))
        color_combo = ttk.Combobox(
            color_row,
            textvariable=self.outline_color_var,
            values=[name for name, _color in OUTLINE_COLOR_CHOICES],
            state="readonly",
            width=12,
        )
        color_combo.pack(side=tk.LEFT)
        color_combo.bind("<<ComboboxSelected>>", lambda _event: self._draw_current_view())
        ttk.Checkbutton(
            color_row,
            text="Bbox",
            variable=self.show_bbox_var,
            command=self._on_overlay_changed,
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(
            color_row,
            text="Caliper",
            variable=self.show_caliper_var,
            command=self._on_overlay_changed,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            color_row,
            text="Hull",
            variable=self.show_hull_var,
            command=self._on_overlay_changed,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            color_row,
            text="Ellipse",
            variable=self.show_ellipse_var,
            command=self._on_overlay_changed,
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.scale_warning_label = tk.Label(
            self.sidebar_content,
            textvariable=self.scale_warning_var,
            fg="#c00000",
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            justify=tk.LEFT,
            wraplength=360,
        )
        self._bind_sidebar_mousewheel(self.scale_warning_label)

        self.summary_card = self._create_card(self.sidebar_content, "Mask summary")
        self.summary_table = ttk.Frame(self.summary_card)
        self.summary_table.pack(fill=tk.X)
        self.details_card = self._create_card(self.sidebar_content, "All available data")
        self.details_table = ttk.Frame(self.details_card)
        self.details_table.pack(fill=tk.X)

        status = ttk.Label(stats_frame, textvariable=self.status_var, anchor="w", padding=(8, 0, 8, 8))
        status.pack(fill=tk.X)

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<ButtonPress-2>", self._on_middle_mouse_down)
        self.canvas.bind("<B2-Motion>", self._on_middle_mouse_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_middle_mouse_up)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        self.root.bind_all("<KeyPress-Shift_L>", self._on_raw_view_down, add="+")
        self.root.bind_all("<KeyPress-Shift_R>", self._on_raw_view_down, add="+")
        self.root.bind_all("<KeyRelease-Shift_L>", self._on_raw_view_up, add="+")
        self.root.bind_all("<KeyRelease-Shift_R>", self._on_raw_view_up, add="+")
        self.root.bind("<FocusOut>", self._on_focus_out)
        self.root.bind("<Left>", lambda _event: self._navigate(-1))
        self.root.bind("<Right>", lambda _event: self._navigate(1))
        self.root.bind("<a>", lambda _event: self._navigate(-1))
        self.root.bind("<A>", lambda _event: self._navigate(-1))
        self.root.bind("<d>", lambda _event: self._navigate(1))
        self.root.bind("<D>", lambda _event: self._navigate(1))
        maximize_tk_window(self.root)
        self.root.after(50, lambda: bring_tk_window_to_front(self.root))

    def _bind_sidebar_mousewheel(self, widget: tk.Widget) -> None:
        widget.bind("<MouseWheel>", self._on_sidebar_mouse_wheel)
        widget.bind("<Button-4>", self._on_sidebar_mouse_wheel)
        widget.bind("<Button-5>", self._on_sidebar_mouse_wheel)

    def _on_sidebar_content_configure(self, _event: tk.Event) -> None:
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def _on_sidebar_canvas_configure(self, event: tk.Event) -> None:
        self.sidebar_canvas.itemconfigure(self.sidebar_window, width=max(1, int(event.width)))
        self._refresh_table_wraplengths()

    def _on_sidebar_mouse_wheel(self, event: tk.Event) -> str:
        delta = int(getattr(event, "delta", 0) or 0)
        if delta != 0:
            units = -1 * (int(delta / 120) if abs(delta) >= 120 else (1 if delta > 0 else -1))
        else:
            button = int(getattr(event, "num", 0) or 0)
            units = -1 if button == 4 else 1 if button == 5 else 0
        if units:
            self.sidebar_canvas.yview_scroll(units * 3, "units")
        return "break"

    def _create_card(self, parent: tk.Widget, title: str) -> ttk.LabelFrame:
        card = ttk.LabelFrame(parent, text=title, padding=8)
        card.pack(fill=tk.X, pady=(0, 10))
        self._bind_sidebar_mousewheel(card)
        return card

    @staticmethod
    def _clear_frame(frame: tk.Widget) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def _value_wraplength(self) -> int:
        width = 360
        if hasattr(self, "sidebar_canvas"):
            width = max(220, int(self.sidebar_canvas.winfo_width()))
        return max(120, width - 180)

    def _refresh_table_wraplengths(self) -> None:
        wraplength = self._value_wraplength()
        if hasattr(self, "scale_warning_label"):
            self.scale_warning_label.configure(wraplength=max(180, wraplength + 140))
        for label in self._table_value_labels:
            try:
                label.configure(wraplength=wraplength)
            except Exception:
                pass

    def _set_scale_warning(self, message: str) -> None:
        self.scale_warning_var.set(message)
        if message:
            if not self.scale_warning_label.winfo_ismapped():
                self.scale_warning_label.pack(fill=tk.X, pady=(0, 10), before=self.summary_card)
        else:
            self.scale_warning_label.pack_forget()

    def _set_table_rows(self, table: tk.Widget, rows: list[tuple[str, Any]]) -> None:
        self._clear_frame(table)
        wraplength = self._value_wraplength()
        for row_index, (field, value) in enumerate(rows):
            row = ttk.Frame(table)
            row.grid(row=row_index, column=0, sticky="ew", pady=(0, 4))
            row.columnconfigure(1, weight=1)
            self._bind_sidebar_mousewheel(row)
            field_label = ttk.Label(row, text=str(field), anchor="w", width=22)
            field_label.grid(row=0, column=0, sticky="nw", padx=(0, 8))
            value_label = ttk.Label(
                row,
                text=format_value(value),
                anchor="w",
                justify=tk.LEFT,
                wraplength=wraplength,
            )
            value_label.grid(row=0, column=1, sticky="ew")
            self._bind_sidebar_mousewheel(field_label)
            self._bind_sidebar_mousewheel(value_label)
            self._table_value_labels.append(value_label)

    def _display_size(self) -> tuple[int, int]:
        if self.current_image is None:
            return (1, 1)
        width, height = self.current_image.size
        return (
            max(1, int(round(width * float(self.zoom_factor)))),
            max(1, int(round(height * float(self.zoom_factor)))),
        )

    def _canvas_size(self) -> tuple[int, int]:
        return (max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))

    def _set_scrollregion(self) -> tuple[int, int]:
        display_width, display_height = self._display_size()
        self.canvas.configure(scrollregion=(0, 0, display_width, display_height))
        return display_width, display_height

    def _canvas_view_xy(self) -> tuple[float, float]:
        return (float(self.canvas.canvasx(0)), float(self.canvas.canvasy(0)))

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _set_canvas_view(self, left: float, top: float, redraw: bool = True) -> None:
        display_width, display_height = self._set_scrollregion()
        canvas_width, canvas_height = self._canvas_size()
        max_left = max(0.0, float(display_width - canvas_width))
        max_top = max(0.0, float(display_height - canvas_height))
        left = self._clamp(float(left), 0.0, max_left)
        top = self._clamp(float(top), 0.0, max_top)
        self.canvas.xview_moveto(0.0 if display_width <= 0 else left / float(display_width))
        self.canvas.yview_moveto(0.0 if display_height <= 0 else top / float(display_height))
        if redraw:
            self._draw_current_view()

    def _canvas_xview(self, *args: Any) -> None:
        self.canvas.xview(*args)
        self._draw_current_view()

    def _canvas_yview(self, *args: Any) -> None:
        self.canvas.yview(*args)
        self._draw_current_view()

    def _on_canvas_configure(self, _event: tk.Event) -> None:
        self._draw_current_view()

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

    def _mouse_wheel_scroll_px(self) -> float:
        canvas_width, canvas_height = self._canvas_size()
        return float(np.clip(max(canvas_width, canvas_height) * 0.08, 40.0, 180.0))

    def _on_mouse_wheel(self, event: tk.Event) -> str:
        steps = self._mouse_wheel_steps(event)
        if steps == 0:
            return "break"
        state = int(getattr(event, "state", 0) or 0)
        shift_down = bool(state & 0x0001)
        ctrl_down = bool(state & 0x0004)
        if ctrl_down:
            self._adjust_zoom(steps, focus_canvas=(float(event.x), float(event.y)))
            return "break"

        scroll_px = self._mouse_wheel_scroll_px()
        if shift_down:
            self._scroll_view(dx=-steps * scroll_px, dy=0.0)
        else:
            self._scroll_view(dx=0.0, dy=-steps * scroll_px)
        return "break"

    def _scroll_view(self, dx: float, dy: float) -> None:
        if self.current_image is None:
            return
        left, top = self._canvas_view_xy()
        self._set_canvas_view(left + float(dx), top + float(dy))

    def _on_middle_mouse_down(self, event: tk.Event) -> str:
        if self.current_image is None:
            self._middle_pan_last_canvas_xy = None
            return "break"
        self._middle_pan_last_canvas_xy = (float(event.x), float(event.y))
        try:
            self.canvas.configure(cursor="fleur")
        except Exception:
            pass
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
        try:
            self.canvas.configure(cursor="")
        except Exception:
            pass
        return "break"

    def _adjust_zoom(self, steps: int, focus_canvas: tuple[float, float] | None = None) -> None:
        if self.current_image is None:
            return
        old_zoom = float(self.zoom_factor)
        canvas_width, canvas_height = self._canvas_size()
        if focus_canvas is None:
            focus_x = canvas_width / 2.0
            focus_y = canvas_height / 2.0
        else:
            focus_x = self._clamp(float(focus_canvas[0]), 0.0, max(0.0, canvas_width - 1.0))
            focus_y = self._clamp(float(focus_canvas[1]), 0.0, max(0.0, canvas_height - 1.0))

        view_left, view_top = self._canvas_view_xy()
        image_focus_x = (view_left + focus_x) / max(old_zoom, 1e-6)
        image_focus_y = (view_top + focus_y) / max(old_zoom, 1e-6)
        updated = float(np.clip(old_zoom * (ZOOM_STEP_FACTOR ** int(steps)), self._zoom_min, self._zoom_max))
        if abs(updated - old_zoom) < 1e-6:
            return
        self.zoom_factor = updated
        new_left = image_focus_x * updated - focus_x
        new_top = image_focus_y * updated - focus_y
        self._set_canvas_view(new_left, new_top)

    def _center_view_on_bbox(self) -> None:
        if self.current_image is None or self.current_bbox is None:
            return
        x0, y0, x1, y1 = self.current_bbox
        zoom = float(self.zoom_factor)
        center_x = ((float(x0) + float(x1) + 1.0) / 2.0) * zoom
        center_y = ((float(y0) + float(y1) + 1.0) / 2.0) * zoom
        canvas_width, canvas_height = self._canvas_size()
        self._set_canvas_view(center_x - canvas_width / 2.0, center_y - canvas_height / 2.0)

    def _schedule_center_view_on_bbox(self) -> None:
        self.root.after_idle(self._center_view_on_bbox)
        self.root.after(200, self._center_view_on_bbox)

    def _ensure_current_caliper(self) -> mask_engine.MaskCaliperResult | None:
        if self.current_binary_mask is None:
            return None
        if self.current_caliper is None:
            self.status_var.set("Computing caliper...")
            self.root.update_idletasks()
            self.current_caliper = mask_engine.mask_longest_caliper(self.current_binary_mask)
        return self.current_caliper

    def _ensure_current_hull(self) -> mask_engine.MaskHullResult | None:
        if self.current_binary_mask is None:
            return None
        if self.current_hull is None:
            self.status_var.set("Computing hull...")
            self.root.update_idletasks()
            self.current_hull = mask_engine.mask_convex_hull(self.current_binary_mask)
        return self.current_hull

    def _ensure_current_ellipse(self) -> mask_engine.MaskEllipseResult | None:
        if self.current_binary_mask is None:
            return None
        if self.current_ellipse is None:
            self.status_var.set("Computing ellipse...")
            self.root.update_idletasks()
            self.current_ellipse = mask_engine.mask_ellipse_fit(self.current_binary_mask)
        return self.current_ellipse

    def _on_overlay_changed(self) -> None:
        if bool(self.show_caliper_var.get()):
            self._ensure_current_caliper()
        if bool(self.show_hull_var.get()):
            self._ensure_current_hull()
        if bool(self.show_ellipse_var.get()):
            self._ensure_current_ellipse()
        if self.current_entry is not None:
            image_size = self.current_image.size if self.current_image is not None else None
            self._populate_stats(
                self.current_entry,
                self.current_record_metadata,
                self.current_bbox,
                image_size,
            )
        self._draw_current_view()

    def _on_raw_view_down(self, _event: tk.Event) -> None:
        if self.raw_view_hold:
            return
        self.raw_view_hold = True
        self._draw_current_view()

    def _on_raw_view_up(self, _event: tk.Event) -> None:
        if not self.raw_view_hold:
            return
        self.raw_view_hold = False
        self._draw_current_view()

    def _on_focus_out(self, _event: tk.Event) -> None:
        if not self.raw_view_hold:
            return
        self.raw_view_hold = False
        self._draw_current_view()

    def _draw_current_view(self) -> None:
        if self.current_image is None or self.current_binary_mask is None or self.current_bbox is None:
            return
        display_width, display_height = self._set_scrollregion()
        canvas_width, canvas_height = self._canvas_size()
        view_xy = self._canvas_view_xy()
        raw_view = bool(self.raw_view_hold)
        caliper_result = None
        hull_result = None
        ellipse_result = None
        if not raw_view and bool(self.show_caliper_var.get()):
            caliper_result = self._ensure_current_caliper()
        if not raw_view and bool(self.show_hull_var.get()):
            hull_result = self._ensure_current_hull()
        if not raw_view and bool(self.show_ellipse_var.get()):
            ellipse_result = self._ensure_current_ellipse()
        preview, origin_xy = build_viewport_image(
            self.current_image,
            self.current_binary_mask,
            self.current_bbox,
            color_for_name(self.outline_color_var.get()),
            self.zoom_factor,
            view_xy,
            (canvas_width, canvas_height),
            show_mask=not raw_view,
            show_bbox=(not raw_view and bool(self.show_bbox_var.get())),
            caliper_result=caliper_result,
            hull_result=hull_result,
            ellipse_result=ellipse_result,
        )
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(origin_xy[0], origin_xy[1], anchor="nw", image=self.preview_photo)
        source_label = self.source_file.name if self.source_file is not None else "annotation metadata"
        width, height = self.current_image.size
        suffix = " | raw" if raw_view else ""
        self.status_var.set(
            f"{source_label} | image {width} x {height} px | zoom {self.zoom_factor:.2f}x | "
            f"view {display_width} x {display_height} px{suffix}"
        )

    def _navigate(self, delta: int) -> None:
        if not self.entries:
            return
        self.index = (self.index + int(delta)) % len(self.entries)
        self._render_current()

    def _open_folder(self) -> None:
        selected = filedialog.askdirectory(
            title="Open SAM mask output folder",
            initialdir=str(self.folder),
            parent=self.root,
        )
        if not selected:
            return
        folder = Path(selected).expanduser().resolve()
        try:
            entries, source_file = load_mask_entries(folder)
        except Exception as exc:
            messagebox.showerror("SAM Mask QC", str(exc), parent=self.root)
            return

        self.folder = folder
        self.entries = entries
        self.source_file = source_file
        self.index = 0
        self.image_cache.clear()
        self.annotation_cache.clear()
        self.preview_photo = None
        self.current_image = None
        self.current_binary_mask = None
        self.current_bbox = None
        self.current_record_metadata = {}
        self.current_entry = None
        self.current_caliper = None
        self.current_hull = None
        self.current_ellipse = None
        self.raw_view_hold = False
        self.root.title(f"SAM Mask QC - {folder.name}")
        self._render_current()
        bring_tk_window_to_front(self.root)

    def _load_image(self, image_path: Path) -> Image.Image:
        image_path = image_path.resolve()
        cached = self.image_cache.get(image_path)
        if cached is not None:
            return cached
        if not image_path.exists():
            raise RuntimeError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        self.image_cache[image_path] = image
        return image

    def _load_annotation(
        self,
        entry: MaskEntry,
        image_shape_hw: tuple[int, int],
    ) -> mask_engine.AnnotationLoadResult:
        key = (
            entry.annotation_path.resolve(),
            entry.metadata_path.resolve(),
            int(image_shape_hw[0]),
            int(image_shape_hw[1]),
        )
        cached = self.annotation_cache.get(key)
        if cached is not None:
            return cached
        if not entry.annotation_path.exists():
            raise RuntimeError(f"Mask archive not found: {entry.annotation_path}")
        loaded = mask_engine.load_annotations(
            entry.annotation_path,
            entry.metadata_path if entry.metadata_path.exists() else None,
            image_shape_hw,
        )
        self.annotation_cache[key] = loaded
        return loaded

    def _record_metadata_for_mask(
        self,
        loaded: mask_engine.AnnotationLoadResult,
        mask_index: int,
    ) -> dict[str, Any]:
        for metadata in loaded.session_metadata.values():
            if parse_int_like(metadata.get("mask_index")) == int(mask_index):
                return dict(metadata)
        records = loaded.payload.get("records", [])
        if isinstance(records, list) and 0 <= int(mask_index) < len(records):
            record = records[int(mask_index)]
            if isinstance(record, dict):
                return dict(record)
        return {}

    def _fallback_threshold(self, loaded: mask_engine.AnnotationLoadResult) -> float:
        for key in ("mask_threshold_applied", "mask_threshold"):
            value = mask_engine.parse_float_like(loaded.payload.get(key))
            if value is not None:
                return float(value)
        return 0.0

    def _bbox_for_entry(
        self,
        entry: MaskEntry,
        record_metadata: dict[str, Any],
        binary_mask: np.ndarray,
        image_size: tuple[int, int],
    ) -> list[int]:
        bbox = (
            parse_bbox(entry.row.get("record_mask_bbox_xyxy"))
            or parse_bbox(entry.row.get("mask_bbox_xyxy"))
            or parse_bbox(record_metadata.get("mask_bbox_xyxy"))
        )
        if bbox is None:
            _area_px, bbox = mask_engine.mask_area_and_bbox(binary_mask)
        if bbox is None:
            raise RuntimeError("Mask has no non-empty bounding box.")
        clamped = clamp_bbox(bbox, image_size[0], image_size[1])
        if clamped is None:
            raise RuntimeError("Mask bounding box is outside the image.")
        return clamped

    def _render_current(self) -> None:
        if not self.entries:
            return
        entry = self.entries[self.index]
        self.position_var.set(f"{self.index + 1} / {len(self.entries)}")
        try:
            image = self._load_image(entry.image_path)
            width, height = image.size
            loaded = self._load_annotation(entry, (height, width))
            if entry.mask_index < 0 or entry.mask_index >= len(loaded.masks):
                raise RuntimeError(
                    f"Mask index {entry.mask_index} is out of range for {entry.image_name}."
                )
            fallback = self._fallback_threshold(loaded)
            binary_mask = mask_engine.binary_committed_mask_at_index(
                loaded.masks,
                entry.mask_index,
                fallback,
                loaded.session_to_mask_index,
                loaded.session_metadata,
                ignore_masks=loaded.ignore_masks,
                ignore_mask_metadata=loaded.ignore_mask_metadata,
            )
            record_metadata = self._record_metadata_for_mask(loaded, entry.mask_index)
            bbox = self._bbox_for_entry(entry, record_metadata, binary_mask, image.size)
            self.current_image = image
            self.current_binary_mask = np.asarray(binary_mask, dtype=bool)
            self.current_bbox = bbox
            self.current_record_metadata = record_metadata
            self.current_entry = entry
            self.current_caliper = None
            self.current_hull = None
            self.current_ellipse = None
            self.raw_view_hold = False
            self.zoom_factor = 1.0
            self._set_scrollregion()
            if bool(self.show_caliper_var.get()):
                self._ensure_current_caliper()
            if bool(self.show_hull_var.get()):
                self._ensure_current_hull()
            if bool(self.show_ellipse_var.get()):
                self._ensure_current_ellipse()
            self._populate_stats(entry, record_metadata, bbox, image.size)
            self._draw_current_view()
            self._schedule_center_view_on_bbox()
        except Exception as exc:
            self.preview_photo = None
            self.current_image = None
            self.current_binary_mask = None
            self.current_bbox = None
            self.current_record_metadata = {}
            self.current_entry = None
            self.current_caliper = None
            self.current_hull = None
            self.current_ellipse = None
            self.raw_view_hold = False
            self.canvas.delete("all")
            self.canvas.configure(scrollregion=(0, 0, 0, 0))
            self.canvas.create_text(
                16,
                16,
                anchor="nw",
                fill="#f0f0f0",
                text=str(exc),
                width=520,
            )
            self._populate_stats(entry, {}, None, None)
            self.status_var.set(str(exc))

    def _populate_stats(
        self,
        entry: MaskEntry,
        record_metadata: dict[str, Any],
        bbox: list[int] | None,
        preview_size: tuple[int, int] | None,
    ) -> None:
        self._table_value_labels = []
        pixel_area = parse_int_like(entry.row.get("pixel_area"))
        if pixel_area is None:
            pixel_area = parse_int_like(entry.row.get("record_mask_area_px"))
        if pixel_area is None:
            pixel_area = parse_int_like(record_metadata.get("mask_area_px"))

        summary_rows: list[tuple[str, Any]] = [
            ("image_name", entry.image_name),
            ("mask_number", entry.mask_number),
        ]
        if pixel_area is not None:
            summary_rows.append(("area_pixels", int(pixel_area)))

        scale = resolve_scale_calibration(self.folder, entry)
        if scale_origin_differs_from_entry(scale, entry):
            self._set_scale_warning("Warning: Scalebar origin image not same as current")
        else:
            self._set_scale_warning("")
        if scale is not None and pixel_area is not None:
            computed_area = int(pixel_area) * scale.units_per_pixel * scale.units_per_pixel
            summary_rows.append((f"area_{scale.length_unit}^2", f"{computed_area:.6g}"))
            summary_rows.append(("units_per_pixel", f"{scale.units_per_pixel:.12g}"))
            if scale.path is not None:
                summary_rows.append(("scale_bar_config", scale.path.name))

        if bool(self.show_caliper_var.get()) and self.current_caliper is not None:
            summary_rows.append(("caliper_pixels", f"{self.current_caliper.distance_px:.3f}"))
            if scale is not None:
                caliper_length = self.current_caliper.distance_px * scale.units_per_pixel
                summary_rows.append((f"caliper_{scale.length_unit}", f"{caliper_length:.6g}"))
        if bool(self.show_hull_var.get()) and self.current_hull is not None:
            summary_rows.append(("hull_area_pixels", f"{self.current_hull.area_px:.3f}"))
            if self.current_hull.solidity is not None:
                summary_rows.append(("solidity", f"{self.current_hull.solidity:.4f}"))
            if scale is not None:
                hull_area = self.current_hull.area_px * scale.units_per_pixel * scale.units_per_pixel
                summary_rows.append((f"hull_area_{scale.length_unit}^2", f"{hull_area:.6g}"))
        if bool(self.show_ellipse_var.get()) and self.current_ellipse is not None:
            summary_rows.append(("ellipse_major_px", f"{self.current_ellipse.major_axis_length_px:.3f}"))
            summary_rows.append(("ellipse_minor_px", f"{self.current_ellipse.minor_axis_length_px:.3f}"))
            summary_rows.append(("ellipse_angle_deg", f"{self.current_ellipse.angle_degrees:.2f}"))
            if scale is not None:
                major_length = self.current_ellipse.major_axis_length_px * scale.units_per_pixel
                minor_length = self.current_ellipse.minor_axis_length_px * scale.units_per_pixel
                summary_rows.append((f"ellipse_major_{scale.length_unit}", f"{major_length:.6g}"))
                summary_rows.append((f"ellipse_minor_{scale.length_unit}", f"{minor_length:.6g}"))

        detail_rows: list[tuple[str, Any]] = []
        seen = {
            "image_name",
            "mask_number",
            "click_number",
            "pixel_area",
            "record_mask_area_px",
            "mask_area_px",
            "computed_area",
            "computed_area_unit",
            "ratio_units_per_pixel",
        }
        for field in entry.field_order:
            if field in seen:
                continue
            value = entry.row.get(field)
            if value is None or str(value) == "":
                continue
            detail_rows.append((field, value))
            seen.add(field)
        for field, value in record_metadata.items():
            prefixed = f"record_{field}"
            if field in seen or prefixed in seen:
                continue
            detail_rows.append((prefixed, value))
            seen.add(prefixed)
        if bbox is not None:
            detail_rows.append(("preview_bbox_xyxy", bbox))
        if preview_size is not None:
            detail_rows.append(("source_image_size_px", [int(preview_size[0]), int(preview_size[1])]))
        if bool(self.show_caliper_var.get()) and self.current_caliper is not None:
            detail_rows.append(("caliper_distance_px", f"{self.current_caliper.distance_px:.3f}"))
            detail_rows.append(("caliper_point_a_xy", list(self.current_caliper.point_a_xy)))
            detail_rows.append(("caliper_point_b_xy", list(self.current_caliper.point_b_xy)))
        if bool(self.show_hull_var.get()) and self.current_hull is not None:
            detail_rows.append(("hull_point_count", len(self.current_hull.points_xy)))
            detail_rows.append(("hull_area_px", f"{self.current_hull.area_px:.3f}"))
            detail_rows.append(("hull_solidity", "" if self.current_hull.solidity is None else f"{self.current_hull.solidity:.6f}"))
            detail_rows.append(("hull_points_xy", [list(point) for point in self.current_hull.points_xy]))
        if bool(self.show_ellipse_var.get()) and self.current_ellipse is not None:
            detail_rows.append(("ellipse_center_xy", [f"{value:.3f}" for value in self.current_ellipse.center_xy]))
            detail_rows.append(("ellipse_major_axis_length_px", f"{self.current_ellipse.major_axis_length_px:.3f}"))
            detail_rows.append(("ellipse_minor_axis_length_px", f"{self.current_ellipse.minor_axis_length_px:.3f}"))
            detail_rows.append(("ellipse_angle_degrees", f"{self.current_ellipse.angle_degrees:.3f}"))
            detail_rows.append(("ellipse_area_px", f"{self.current_ellipse.area_px:.3f}"))
            detail_rows.append(("ellipse_eccentricity", f"{self.current_ellipse.eccentricity:.6f}"))

        self._set_table_rows(self.summary_table, summary_rows)
        self._set_table_rows(self.details_table, detail_rows)
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

def choose_source_folder() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askdirectory(title="Select SAM mask output folder")
    root.destroy()
    if not selected:
        return None
    return Path(selected)


def main() -> int:
    args = parse_args()
    raw_folder = args.source_folder_option or args.source_folder
    folder = Path(raw_folder) if raw_folder else choose_source_folder()
    if folder is None:
        return 0
    output_file = Path(args.output_file) if args.output_file else None
    try:
        entries, source_file = load_mask_entries(folder, output_file)
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("SAM Mask QC", str(exc))
            root.destroy()
        except Exception:
            pass
        return 1

    root = tk.Tk()
    SamMaskQcApp(root, Path(folder).expanduser().resolve(), entries, source_file)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
