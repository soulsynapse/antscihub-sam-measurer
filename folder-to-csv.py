#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


ANNOTATION_METADATA_SUFFIX = ".sam_clicks.json"
SCALE_BAR_CONFIG_SUFFIX = ".scale_bar_config.result.json"
DEFAULT_OUTPUT_CSV_NAME = "sam_mask_measurements.csv"


BASE_CSV_COLUMNS = [
    "image_name",
    "click_number",
    "pixel_area",
    "ratio_units_per_pixel",
    "computed_area",
    "computed_area_unit",
]


def sanitize_column_part(raw: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", raw.strip())
    cleaned = cleaned.strip("_").lower()
    return cleaned or "value"


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def flatten_json_fields(prefix: str, value: Any) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if not isinstance(value, dict):
        flattened[sanitize_column_part(prefix)] = csv_value(value)
        return flattened

    for key, child in value.items():
        column = f"{sanitize_column_part(prefix)}_{sanitize_column_part(str(key))}"
        if isinstance(child, dict):
            flattened.update(flatten_json_fields(column, child))
        else:
            flattened[column] = csv_value(child)
    return flattened


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export SAM click-mask areas from a folder to CSV using the folder's "
            "scale-bar calibration JSON."
        ),
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help="Image folder to export. If omitted, a folder picker is shown.",
    )
    parser.add_argument(
        "--source-folder",
        "--folder",
        dest="source_folder",
        help="Image folder to export. Overrides the positional folder.",
    )
    parser.add_argument(
        "--scale-bar-config",
        dest="scale_bar_config",
        help=(
            "Optional explicit scale-bar JSON. Defaults to the newest "
            f"*{SCALE_BAR_CONFIG_SUFFIX} file in the selected folder."
        ),
    )
    parser.add_argument(
        "--output-csv",
        dest="output_csv",
        help=(
            "Optional output CSV path. Defaults to "
            f"{DEFAULT_OUTPUT_CSV_NAME} in the selected folder."
        ),
    )
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON file must contain an object: {path}")
    return payload


def select_folder_with_gui() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("No folder was provided and tkinter is unavailable.") from exc

    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    selected = filedialog.askdirectory(title="Select image folder to export")
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


def resolve_scale_bar_config(folder: Path, raw_path: str | None) -> Path:
    if raw_path:
        path = Path(str(raw_path)).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"Scale-bar config does not exist: {path}")
        if not path.is_file():
            raise RuntimeError(f"Scale-bar config is not a file: {path}")
        return path

    candidates = sorted(
        folder.glob(f"*{SCALE_BAR_CONFIG_SUFFIX}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(
            f"No *{SCALE_BAR_CONFIG_SUFFIX} file found in selected folder: {folder}"
        )
    if len(candidates) > 1:
        print(
            "Multiple scale-bar config files found; using newest: "
            f"{candidates[0].name}",
            file=sys.stderr,
        )
    return candidates[0]


def read_scale_bar_config(scale_bar_config: Path) -> tuple[dict[str, Any], float, str]:
    payload = load_json_object(scale_bar_config)
    scale_bar = payload.get("scale_bar")
    if not isinstance(scale_bar, dict):
        raise RuntimeError(f"Scale-bar config is missing scale_bar object: {scale_bar_config}")

    try:
        units_per_pixel = float(scale_bar["units_per_pixel"])
    except Exception as exc:
        raise RuntimeError(
            f"Scale-bar config is missing numeric scale_bar.units_per_pixel: {scale_bar_config}"
        ) from exc

    if units_per_pixel <= 0:
        raise RuntimeError(
            f"scale_bar.units_per_pixel must be greater than zero: {scale_bar_config}"
        )

    length_unit = str(scale_bar.get("length_unit") or "units").strip() or "units"
    return payload, units_per_pixel, length_unit


def image_name_from_annotation(path: Path, payload: dict[str, Any]) -> str:
    image_name = str(payload.get("image_name") or "").strip()
    if image_name:
        return image_name

    name = path.name
    if name.endswith(ANNOTATION_METADATA_SUFFIX):
        return name[: -len(ANNOTATION_METADATA_SUFFIX)]
    return path.stem


def read_area_rows(
    folder: Path,
    scale_bar_config: Path,
    scale_payload: dict[str, Any],
    units_per_pixel: float,
    length_unit: str,
) -> list[dict[str, Any]]:
    annotation_paths = sorted(folder.glob(f"*{ANNOTATION_METADATA_SUFFIX}"))
    if not annotation_paths:
        raise RuntimeError(
            f"No *{ANNOTATION_METADATA_SUFFIX} files found in selected folder: {folder}"
        )

    rows: list[dict[str, Any]] = []
    area_unit = f"{length_unit}^2"
    area_ratio = units_per_pixel * units_per_pixel
    scale_metadata = {
        "scale_bar_config_path": str(scale_bar_config),
        "scale_bar_config_name": scale_bar_config.name,
        **flatten_json_fields("scale_config", scale_payload),
    }

    for annotation_path in annotation_paths:
        payload = load_json_object(annotation_path)
        image_name = image_name_from_annotation(annotation_path, payload)
        annotation_metadata_payload = {
            key: value for key, value in payload.items() if key != "records"
        }
        annotation_metadata = {
            "annotation_metadata_path": str(annotation_path),
            "annotation_metadata_name": annotation_path.name,
            **flatten_json_fields("annotation", annotation_metadata_payload),
        }
        records = payload.get("records", [])
        if not isinstance(records, list):
            continue

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue

            raw_area = record.get("mask_area_px")
            if raw_area is None:
                continue

            try:
                pixel_area = int(round(float(raw_area)))
            except (TypeError, ValueError):
                continue

            raw_mask_index = record.get("mask_index", index)
            try:
                click_number = int(raw_mask_index) + 1
            except (TypeError, ValueError):
                click_number = index + 1

            rows.append(
                {
                    "image_name": image_name,
                    "click_number": click_number,
                    "pixel_area": pixel_area,
                    "ratio_units_per_pixel": units_per_pixel,
                    "computed_area": pixel_area * area_ratio,
                    "computed_area_unit": area_unit,
                    **scale_metadata,
                    **annotation_metadata,
                    **flatten_json_fields("record", record),
                }
            )

    if not rows:
        raise RuntimeError(
            "No mask area records were found in the annotation metadata files."
        )
    return rows


def resolve_output_csv(folder: Path, raw_path: str | None) -> Path:
    if raw_path:
        return Path(str(raw_path)).expanduser().resolve()
    return folder / DEFAULT_OUTPUT_CSV_NAME


def write_csv(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BASE_CSV_COLUMNS)
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    folder = resolve_source_folder(args)
    scale_bar_config = resolve_scale_bar_config(folder, args.scale_bar_config)
    scale_payload, units_per_pixel, length_unit = read_scale_bar_config(scale_bar_config)
    rows = read_area_rows(
        folder=folder,
        scale_bar_config=scale_bar_config,
        scale_payload=scale_payload,
        units_per_pixel=units_per_pixel,
        length_unit=length_unit,
    )
    output_csv = resolve_output_csv(folder, args.output_csv)
    write_csv(output_csv, rows)

    print(f"Wrote CSV: {output_csv}", flush=True)
    print(f"Rows: {len(rows)}", flush=True)
    print(f"Scale-bar config: {scale_bar_config}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
