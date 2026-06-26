from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AnnotationLoadResult:
    masks: list[np.ndarray]
    ignore_masks: list[np.ndarray]
    session_to_mask_index: dict[int, int]
    session_metadata: dict[int, dict[str, Any]]
    ignore_mask_metadata: list[dict[str, Any]]
    active_session_id: int
    payload: dict[str, Any]


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


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def load_json_object_optional(path: Path | None) -> dict[str, Any]:
    if path is None or (not path.exists()):
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def metadata_json_to_payload(metadata_json: Any) -> dict[str, Any]:
    if metadata_json is None:
        return {}
    try:
        if np.ndim(metadata_json) == 0:
            metadata_text = str(metadata_json.item())
        else:
            metadata_text = str(np.asarray(metadata_json).reshape(-1)[0])
        if not metadata_text:
            return {}
        loaded_payload = json.loads(metadata_text)
    except Exception:
        return {}
    if isinstance(loaded_payload, dict):
        return loaded_payload
    return {}


def metadata_marks_binary_mask(md: dict[str, Any]) -> bool:
    for key in ("mask_values_are_binary", "stored_mask_values_are_binary"):
        parsed = parse_bool_like(md.get(key))
        if parsed is True:
            return True
    kind = str(md.get("mask_values_kind", "")).strip().lower()
    stored_kind = str(md.get("stored_mask_values_kind", "")).strip().lower()
    return kind in {"binary", "thresholded_binary"} or stored_kind in {
        "binary",
        "thresholded_binary",
    }


def array_values_are_binary(mask: np.ndarray) -> bool:
    arr = np.asarray(mask)
    if arr.size == 0:
        return False
    return bool(np.all((arr == 0) | (arr == 1) | (arr == 255)))


def threshold_from_metadata(md: dict[str, Any], fallback: float) -> float:
    if metadata_marks_binary_mask(md):
        return 0.5
    for key in ("threshold_last_saved", "threshold_at_click"):
        parsed = parse_float_like(md.get(key))
        if parsed is not None:
            return float(parsed)
    return float(fallback)


def sessions_by_mask_index(
    session_to_mask_index: dict[int, int],
    committed_mask_count: int,
) -> dict[int, int]:
    sessions_by_index: dict[int, int] = {}
    for sid, idx in session_to_mask_index.items():
        if 0 <= int(idx) < int(committed_mask_count):
            sessions_by_index[int(idx)] = int(sid)
    return sessions_by_index


def session_metadata_for_mask_index(
    idx: int,
    session_to_mask_index: dict[int, int],
    session_metadata: dict[int, dict[str, Any]],
    committed_mask_count: int,
) -> dict[str, Any]:
    by_index = sessions_by_mask_index(session_to_mask_index, committed_mask_count)
    sid = by_index.get(int(idx))
    if sid is None:
        return {}
    return session_metadata.get(int(sid), {})


def threshold_for_committed_mask(
    idx: int,
    fallback: float,
    session_to_mask_index: dict[int, int],
    session_metadata: dict[int, dict[str, Any]],
    committed_mask_count: int,
) -> float:
    return threshold_from_metadata(
        session_metadata_for_mask_index(
            idx,
            session_to_mask_index,
            session_metadata,
            committed_mask_count,
        ),
        fallback,
    )


def threshold_for_ignore_mask(
    idx: int,
    fallback: float,
    ignore_mask_metadata: list[dict[str, Any]],
) -> float:
    md = ignore_mask_metadata[idx] if 0 <= int(idx) < len(ignore_mask_metadata) else {}
    return threshold_from_metadata(md, fallback)


def thresholds_for_committed_masks(
    committed_masks: list[np.ndarray],
    fallback: float,
    session_to_mask_index: dict[int, int],
    session_metadata: dict[int, dict[str, Any]],
) -> list[float]:
    return [
        threshold_for_committed_mask(
            idx,
            fallback,
            session_to_mask_index,
            session_metadata,
            len(committed_masks),
        )
        for idx in range(len(committed_masks))
    ]


def thresholds_for_ignore_masks(
    ignore_masks: list[np.ndarray],
    fallback: float,
    ignore_mask_metadata: list[dict[str, Any]],
) -> list[float]:
    return [
        threshold_for_ignore_mask(idx, fallback, ignore_mask_metadata)
        for idx in range(len(ignore_masks))
    ]


def binary_ignore_mask_at_index(
    ignore_masks: list[np.ndarray],
    ignore_mask_metadata: list[dict[str, Any]],
    idx: int,
    fallback: float,
) -> np.ndarray:
    mask = np.asarray(ignore_masks[idx], dtype=np.float32)
    threshold = threshold_for_ignore_mask(idx, fallback, ignore_mask_metadata)
    return np.asarray(mask > threshold)


def combined_ignore_mask(
    ignore_masks: list[np.ndarray],
    ignore_mask_metadata: list[dict[str, Any]],
    fallback: float,
) -> np.ndarray | None:
    if not ignore_masks:
        return None
    combined: np.ndarray | None = None
    for idx in range(len(ignore_masks)):
        mask = binary_ignore_mask_at_index(ignore_masks, ignore_mask_metadata, idx, fallback)
        if mask.ndim != 2:
            continue
        if combined is None:
            combined = np.zeros(mask.shape, dtype=bool)
        if combined.shape != mask.shape:
            continue
        combined |= mask
    return combined


def binary_mask_with_ignore(
    mask: np.ndarray,
    threshold: float,
    ignore_masks: list[np.ndarray] | None = None,
    ignore_mask_metadata: list[dict[str, Any]] | None = None,
    subtract_ignored: bool = True,
) -> np.ndarray:
    binary = np.asarray(np.asarray(mask, dtype=np.float32) > float(threshold))
    if subtract_ignored:
        ignore_mask = combined_ignore_mask(
            list(ignore_masks or []),
            list(ignore_mask_metadata or []),
            float(threshold),
        )
        if ignore_mask is not None and ignore_mask.shape == binary.shape:
            binary = binary & (~ignore_mask)
    return binary


def binary_committed_mask_at_index(
    committed_masks: list[np.ndarray],
    idx: int,
    fallback: float,
    session_to_mask_index: dict[int, int],
    session_metadata: dict[int, dict[str, Any]],
    ignore_masks: list[np.ndarray] | None = None,
    ignore_mask_metadata: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    mask = np.asarray(committed_masks[idx], dtype=np.float32)
    threshold = threshold_for_committed_mask(
        idx,
        fallback,
        session_to_mask_index,
        session_metadata,
        len(committed_masks),
    )
    return binary_mask_with_ignore(
        mask,
        threshold,
        ignore_masks=ignore_masks,
        ignore_mask_metadata=ignore_mask_metadata,
        subtract_ignored=True,
    )


def mask_area_and_bbox(binary: np.ndarray) -> tuple[int, list[int] | None]:
    mask = np.asarray(binary, dtype=bool)
    area_px = int(np.count_nonzero(mask))
    if area_px <= 0:
        return 0, None
    ys, xs = np.nonzero(mask)
    return area_px, [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_metadata_fields(mask: np.ndarray, threshold: float) -> dict[str, Any]:
    mask_float = np.asarray(mask, dtype=np.float32)
    binary = np.asarray(mask_float > float(threshold))
    area_px, bbox_xyxy = mask_area_and_bbox(binary)
    return {
        "mask_area_px": int(area_px),
        "mask_bbox_xyxy": bbox_xyxy,
        "mask_logit_min": float(np.min(mask_float)),
        "mask_logit_max": float(np.max(mask_float)),
        "mask_logit_mean": float(np.mean(mask_float)),
    }


def build_ignore_mask_metadata(
    ignore_index: int,
    seed_point: tuple[float, float],
    mask: np.ndarray,
    threshold: float,
    continuous_mask_only: bool,
    saved_at: str,
    image_shape_hw: tuple[int, int],
    model_name: str,
) -> dict[str, Any]:
    h, w = image_shape_hw
    return {
        "kind": "ignore",
        "ignore_index": int(ignore_index),
        "seed_point_xy": [float(seed_point[0]), float(seed_point[1])],
        "seed_point_xy_round": [int(round(seed_point[0])), int(round(seed_point[1]))],
        "created_at_utc": saved_at,
        "last_updated_utc": saved_at,
        "image_size_hw": [int(h), int(w)],
        "model_name": str(model_name),
        "threshold_at_click": float(threshold),
        "threshold_last_saved": float(threshold),
        "continuous_mask_only_at_click": bool(continuous_mask_only),
        "mask_values_are_binary": False,
        "mask_values_kind": "sam_logits",
        **mask_metadata_fields(mask, threshold),
    }


def build_annotation_payload(
    saved_at: str,
    image_path: Path | None,
    image_shape_hw: tuple[int, int],
    model_name: str,
    mask_threshold: float,
    continuous_mask_only: bool,
    committed_masks: list[np.ndarray],
    ignore_masks: list[np.ndarray],
    session_to_mask_index: dict[int, int],
    session_metadata: dict[int, dict[str, Any]],
    ignore_mask_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    h, w = image_shape_hw
    by_index = sessions_by_mask_index(session_to_mask_index, len(committed_masks))
    threshold_for_export = float(mask_threshold)

    records: list[dict[str, Any]] = []
    total_mask_area_px = 0
    for idx in range(len(committed_masks)):
        sid = int(by_index.get(idx, idx + 1))
        md = dict(session_metadata.get(sid, {}))
        md["session_id"] = sid
        md["mask_index"] = int(idx)
        md["last_updated_utc"] = md.get("last_updated_utc", saved_at)
        binary = binary_committed_mask_at_index(
            committed_masks,
            idx,
            threshold_for_export,
            session_to_mask_index,
            session_metadata,
            ignore_masks=ignore_masks,
            ignore_mask_metadata=ignore_mask_metadata,
        )
        area_px, bbox_xyxy = mask_area_and_bbox(binary)
        total_mask_area_px += int(area_px)
        md["mask_area_px"] = int(area_px)
        md["mask_bbox_xyxy"] = bbox_xyxy
        md["stored_mask_values_are_binary"] = True
        md["stored_mask_values_kind"] = "thresholded_binary"
        md["ignore_regions_applied"] = int(len(ignore_masks))
        records.append(md)

    ignore_records: list[dict[str, Any]] = []
    ignore_total_mask_area_px = 0
    for idx in range(len(ignore_masks)):
        md = dict(ignore_mask_metadata[idx]) if idx < len(ignore_mask_metadata) else {}
        md["ignore_index"] = int(idx)
        md["last_updated_utc"] = md.get("last_updated_utc", saved_at)
        binary = binary_ignore_mask_at_index(
            ignore_masks,
            ignore_mask_metadata,
            idx,
            threshold_for_export,
        )
        area_px, bbox_xyxy = mask_area_and_bbox(binary)
        ignore_total_mask_area_px += int(area_px)
        md["mask_area_px"] = int(area_px)
        md["mask_bbox_xyxy"] = bbox_xyxy
        md["stored_mask_values_are_binary"] = True
        md["stored_mask_values_kind"] = "thresholded_binary"
        ignore_records.append(md)

    return {
        "version": 1,
        "image_path": str(image_path) if image_path is not None else "",
        "image_name": image_path.name if image_path is not None else "",
        "image_size_hw": [int(h), int(w)],
        "saved_at_utc": saved_at,
        "model_name": str(model_name),
        "mask_threshold": float(mask_threshold),
        "continuous_mask_only": bool(continuous_mask_only),
        "stored_mask_values_kind": "thresholded_binary",
        "mask_count": int(len(records)),
        "ignore_mask_count": int(len(ignore_records)),
        "total_mask_area_px": int(total_mask_area_px),
        "ignore_total_mask_area_px": int(ignore_total_mask_area_px),
        "records": records,
        "ignore_records": ignore_records,
    }


def save_annotations(
    save_path: Path,
    metadata_path: Path | None,
    image_path: Path | None,
    image_shape_hw: tuple[int, int],
    model_name: str,
    mask_threshold: float,
    continuous_mask_only: bool,
    committed_masks: list[np.ndarray],
    ignore_masks: list[np.ndarray],
    session_to_mask_index: dict[int, int],
    session_metadata: dict[int, dict[str, Any]],
    ignore_mask_metadata: list[dict[str, Any]],
    saved_at: str,
) -> dict[str, Any] | None:
    if not committed_masks and not ignore_masks:
        if save_path.exists():
            save_path.unlink()
        if metadata_path is not None and metadata_path.exists():
            metadata_path.unlink()
        return None

    h, w = image_shape_hw
    threshold_for_export = float(mask_threshold)
    thresholds_for_export = thresholds_for_committed_masks(
        committed_masks,
        threshold_for_export,
        session_to_mask_index,
        session_metadata,
    )
    ignore_thresholds_for_export = thresholds_for_ignore_masks(
        ignore_masks,
        threshold_for_export,
        ignore_mask_metadata,
    )
    if committed_masks:
        binary_masks = np.stack(
            [
                np.asarray(
                    binary_mask_with_ignore(
                        committed_masks[idx],
                        thresholds_for_export[idx],
                        ignore_masks=ignore_masks,
                        ignore_mask_metadata=ignore_mask_metadata,
                        subtract_ignored=False,
                    ),
                    dtype=np.uint8,
                )
                for idx in range(len(committed_masks))
            ],
            axis=0,
        )
    else:
        binary_masks = np.zeros((0, h, w), dtype=np.uint8)

    if ignore_masks:
        ignore_binary_masks = np.stack(
            [
                np.asarray(
                    binary_ignore_mask_at_index(
                        ignore_masks,
                        ignore_mask_metadata,
                        idx,
                        threshold_for_export,
                    ),
                    dtype=np.uint8,
                )
                for idx in range(len(ignore_masks))
            ],
            axis=0,
        )
    else:
        ignore_binary_masks = np.zeros((0, h, w), dtype=np.uint8)

    packed_masks = np.packbits(binary_masks, axis=2, bitorder="little")
    ignore_packed_masks = np.packbits(ignore_binary_masks, axis=2, bitorder="little")
    payload = build_annotation_payload(
        saved_at=saved_at,
        image_path=image_path,
        image_shape_hw=image_shape_hw,
        model_name=model_name,
        mask_threshold=threshold_for_export,
        continuous_mask_only=continuous_mask_only,
        committed_masks=committed_masks,
        ignore_masks=ignore_masks,
        session_to_mask_index=session_to_mask_index,
        session_metadata=session_metadata,
        ignore_mask_metadata=ignore_mask_metadata,
    )
    payload["mask_encoding"] = "bitpack_binary_v1"
    payload["mask_threshold_mode"] = "per_mask"
    payload["mask_threshold_applied"] = threshold_for_export
    payload["mask_thresholds_applied"] = [float(value) for value in thresholds_for_export]
    payload["ignore_mask_thresholds_applied"] = [
        float(value) for value in ignore_thresholds_for_export
    ]
    payload["mask_shape_nhw"] = [
        int(binary_masks.shape[0]),
        int(binary_masks.shape[1]),
        int(binary_masks.shape[2]),
    ]
    payload["ignore_mask_shape_nhw"] = [
        int(ignore_binary_masks.shape[0]),
        int(ignore_binary_masks.shape[1]),
        int(ignore_binary_masks.shape[2]),
    ]
    payload["packed_width_bytes"] = int(packed_masks.shape[2])
    payload["ignore_packed_width_bytes"] = int(ignore_packed_masks.shape[2])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            masks_packed=np.asarray(packed_masks, dtype=np.uint8),
            masks_shape_nhw=np.asarray(binary_masks.shape, dtype=np.int32),
            ignore_masks_packed=np.asarray(ignore_packed_masks, dtype=np.uint8),
            ignore_masks_shape_nhw=np.asarray(ignore_binary_masks.shape, dtype=np.int32),
            metadata_json=np.asarray(json.dumps(payload), dtype=np.str_),
        )
    tmp_path.replace(save_path)
    if metadata_path is not None:
        write_json_atomic(metadata_path, payload)
    return payload


def unpack_packed_masks(
    packed: np.ndarray,
    shape_arr: np.ndarray,
    label: str,
) -> np.ndarray:
    if packed.ndim != 3:
        raise RuntimeError(f"unexpected {label} packed mask shape")
    if shape_arr.size != 3:
        raise RuntimeError(f"missing shape metadata for {label} packed masks")
    n, mh, mw = (int(shape_arr[0]), int(shape_arr[1]), int(shape_arr[2]))
    if n < 0 or mh <= 0 or mw <= 0:
        raise RuntimeError(f"invalid saved {label} packed mask dimensions")
    if n != int(packed.shape[0]) or mh != int(packed.shape[1]):
        raise RuntimeError(f"{label} packed mask dimensions do not match shape metadata")
    if mw > int(packed.shape[2]) * 8:
        raise RuntimeError(f"{label} packed mask width metadata exceeds packed payload width")
    return np.unpackbits(
        packed,
        axis=2,
        count=mw,
        bitorder="little",
    ).astype(np.float32)


def load_annotations(
    save_path: Path,
    metadata_path: Path | None,
    image_shape_hw: tuple[int, int],
) -> AnnotationLoadResult:
    loaded_masks_are_binary = False
    loaded_ignore_masks_are_binary = False
    with np.load(save_path, allow_pickle=False) as data:
        metadata_json = data["metadata_json"] if "metadata_json" in data.files else None
        if "masks_packed" in data.files:
            loaded_masks_are_binary = True
            masks = unpack_packed_masks(
                np.asarray(data["masks_packed"], dtype=np.uint8),
                np.asarray(data["masks_shape_nhw"], dtype=np.int64).reshape(-1)
                if "masks_shape_nhw" in data.files
                else np.asarray([], dtype=np.int64),
                "saved",
            )
        elif "masks" in data.files:
            masks = np.asarray(data["masks"])
        else:
            raise RuntimeError("saved annotations missing mask payload")

        if "ignore_masks_packed" in data.files:
            loaded_ignore_masks_are_binary = True
            ignore_masks = unpack_packed_masks(
                np.asarray(data["ignore_masks_packed"], dtype=np.uint8),
                np.asarray(data["ignore_masks_shape_nhw"], dtype=np.int64).reshape(-1)
                if "ignore_masks_shape_nhw" in data.files
                else np.asarray([], dtype=np.int64),
                "ignore",
            )
        elif "ignore_masks" in data.files:
            ignore_masks = np.asarray(data["ignore_masks"])
        else:
            ignore_masks = np.zeros((0, 1, 1), dtype=np.float32)

    if masks.ndim != 3:
        raise RuntimeError("unexpected mask shape")
    if ignore_masks.ndim != 3:
        raise RuntimeError("unexpected ignore mask shape")

    h, w = image_shape_hw
    if tuple(masks.shape[1:]) != (h, w):
        raise RuntimeError("mask size mismatch")
    if int(ignore_masks.shape[0]) == 0:
        ignore_masks = np.zeros((0, h, w), dtype=np.float32)
    elif tuple(ignore_masks.shape[1:]) != (h, w):
        raise RuntimeError("ignore mask size mismatch")

    payload = metadata_json_to_payload(metadata_json)
    if not payload:
        payload = load_json_object_optional(metadata_path)

    records: list[Any] = []
    loaded_records = payload.get("records", [])
    if isinstance(loaded_records, list):
        records = loaded_records
    ignore_records: list[Any] = []
    loaded_ignore_records = payload.get("ignore_records", [])
    if isinstance(loaded_ignore_records, list):
        ignore_records = loaded_ignore_records
    payload_marks_binary = metadata_marks_binary_mask(payload)
    thresholds_applied_raw = payload.get("mask_thresholds_applied", [])
    thresholds_applied = thresholds_applied_raw if isinstance(thresholds_applied_raw, list) else []
    ignore_thresholds_applied_raw = payload.get("ignore_mask_thresholds_applied", [])
    ignore_thresholds_applied = (
        ignore_thresholds_applied_raw
        if isinstance(ignore_thresholds_applied_raw, list)
        else []
    )

    committed_mask_list: list[np.ndarray] = []
    ignore_mask_list: list[np.ndarray] = []
    session_to_mask_index: dict[int, int] = {}
    session_metadata: dict[int, dict[str, Any]] = {}
    ignore_mask_metadata: list[dict[str, Any]] = []

    for idx in range(int(masks.shape[0])):
        committed_mask_list.append(np.asarray(masks[idx], dtype=np.float32))
        rec = records[idx] if idx < len(records) and isinstance(records[idx], dict) else {}
        sid_raw = rec.get("session_id", idx + 1)
        try:
            session_id = int(sid_raw)
        except Exception:
            session_id = idx + 1
        while session_id <= 0 or session_id in session_to_mask_index:
            session_id += 1
        rec_data = dict(rec)
        rec_data["session_id"] = int(session_id)
        rec_data["mask_index"] = int(idx)

        threshold_applied = None
        if idx < len(thresholds_applied):
            threshold_applied = parse_float_like(thresholds_applied[idx])
        if threshold_applied is None:
            threshold_applied = parse_float_like(payload.get("mask_threshold_applied"))
        if threshold_applied is None:
            threshold_applied = parse_float_like(payload.get("mask_threshold"))
        if threshold_applied is not None:
            rec_data.setdefault("threshold_last_saved", float(threshold_applied))
            rec_data.setdefault("threshold_at_click", float(threshold_applied))
            rec_data.setdefault("threshold_applied_to_saved_mask", float(threshold_applied))

        mask_values_are_binary = bool(
            loaded_masks_are_binary
            or payload_marks_binary
            or metadata_marks_binary_mask(rec_data)
            or array_values_are_binary(masks[idx])
        )
        if mask_values_are_binary:
            rec_data["mask_values_are_binary"] = True
            rec_data["mask_values_kind"] = "thresholded_binary"
            binary = np.asarray(masks[idx] > 0.5)
            area_px, bbox_xyxy = mask_area_and_bbox(binary)
            rec_data.setdefault("mask_area_px", int(area_px))
            rec_data.setdefault("mask_bbox_xyxy", bbox_xyxy)
        else:
            rec_data["mask_values_are_binary"] = False
            rec_data.setdefault("mask_values_kind", "sam_logits")

        session_to_mask_index[int(session_id)] = int(idx)
        session_metadata[int(session_id)] = rec_data

    for idx in range(int(ignore_masks.shape[0])):
        ignore_mask_list.append(np.asarray(ignore_masks[idx], dtype=np.float32))
        rec = (
            ignore_records[idx]
            if idx < len(ignore_records) and isinstance(ignore_records[idx], dict)
            else {}
        )
        rec_data = dict(rec)
        rec_data["ignore_index"] = int(idx)

        threshold_applied = None
        if idx < len(ignore_thresholds_applied):
            threshold_applied = parse_float_like(ignore_thresholds_applied[idx])
        if threshold_applied is None:
            threshold_applied = parse_float_like(payload.get("mask_threshold_applied"))
        if threshold_applied is None:
            threshold_applied = parse_float_like(payload.get("mask_threshold"))
        if threshold_applied is not None:
            rec_data.setdefault("threshold_last_saved", float(threshold_applied))
            rec_data.setdefault("threshold_at_click", float(threshold_applied))
            rec_data.setdefault("threshold_applied_to_saved_mask", float(threshold_applied))

        mask_values_are_binary = bool(
            loaded_ignore_masks_are_binary
            or payload_marks_binary
            or metadata_marks_binary_mask(rec_data)
            or array_values_are_binary(ignore_masks[idx])
        )
        if mask_values_are_binary:
            rec_data["mask_values_are_binary"] = True
            rec_data["mask_values_kind"] = "thresholded_binary"
            binary = np.asarray(ignore_masks[idx] > 0.5)
            area_px, bbox_xyxy = mask_area_and_bbox(binary)
            rec_data.setdefault("mask_area_px", int(area_px))
            rec_data.setdefault("mask_bbox_xyxy", bbox_xyxy)
        else:
            rec_data["mask_values_are_binary"] = False
            rec_data.setdefault("mask_values_kind", "sam_logits")
        ignore_mask_metadata.append(rec_data)

    max_session = max(session_to_mask_index.keys(), default=0)
    active_session_id = int(max_session + 1) if max_session > 0 else 0
    return AnnotationLoadResult(
        masks=committed_mask_list,
        ignore_masks=ignore_mask_list,
        session_to_mask_index=session_to_mask_index,
        session_metadata=session_metadata,
        ignore_mask_metadata=ignore_mask_metadata,
        active_session_id=active_session_id,
        payload=payload,
    )