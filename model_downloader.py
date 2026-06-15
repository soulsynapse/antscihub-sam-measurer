from __future__ import annotations

from pathlib import Path

from onnx_dependencies import MODELS, ensure_model_weights, get_model


DEFAULT_MODEL_NAME = "Segment-Anything (accuracy)"


def available_model_names() -> list[str]:
    return [model.name for model in MODELS]


def model_weight_filenames(model_name: str) -> tuple[str, str]:
    model = get_model(model_name)
    encoder_name = Path(model.encoder_weight.url).name
    decoder_name = Path(model.decoder_weight.url).name
    return encoder_name, decoder_name


def model_weight_paths(model_name: str, models_dir: str | Path) -> tuple[Path, Path]:
    base = Path(models_dir)
    encoder_name, decoder_name = model_weight_filenames(model_name)
    return base / encoder_name, base / decoder_name


def download_selected_model(
    model_name: str,
    models_dir: str | Path,
) -> tuple[Path, Path]:
    return ensure_model_weights(model_name=model_name, models_dir=models_dir)
