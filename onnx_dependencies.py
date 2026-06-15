from __future__ import annotations

import collections
import inspect
from pathlib import Path

import gdown

Model = collections.namedtuple("Model", ["name", "encoder_weight", "decoder_weight"])
Weight = collections.namedtuple("Weight", ["url", "md5"])

# Mirrors Annolid's SAM model registry/checksum approach for ONNX assets.
MODELS = [
    Model(
        name="Segment-Anything (Edge)",
        encoder_weight=Weight(
            url=(
                "https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/"
                "edge_sam_3x_encoder.onnx"
            ),
            md5="e0745d06f3ee9c5e01a667b56a40875b",
        ),
        decoder_weight=Weight(
            url=(
                "https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/"
                "edge_sam_3x_decoder.onnx"
            ),
            md5="9fe1d5521b4349ab710e9cc970936970",
        ),
    ),
    Model(
        name="Segment-Anything (speed)",
        encoder_weight=Weight(
            url=(
                "https://github.com/wkentaro/labelme/releases/download/sam-20230416/"
                "sam_vit_b_01ec64.quantized.encoder.onnx"
            ),
            md5="80fd8d0ab6c6ae8cb7b3bd5f368a752c",
        ),
        decoder_weight=Weight(
            url=(
                "https://github.com/wkentaro/labelme/releases/download/sam-20230416/"
                "sam_vit_b_01ec64.quantized.decoder.onnx"
            ),
            md5="4253558be238c15fc265a7a876aaec82",
        ),
    ),
    Model(
        name="Segment-Anything (balanced)",
        encoder_weight=Weight(
            url=(
                "https://github.com/wkentaro/labelme/releases/download/sam-20230416/"
                "sam_vit_l_0b3195.quantized.encoder.onnx"
            ),
            md5="080004dc9992724d360a49399d1ee24b",
        ),
        decoder_weight=Weight(
            url=(
                "https://github.com/wkentaro/labelme/releases/download/sam-20230416/"
                "sam_vit_l_0b3195.quantized.decoder.onnx"
            ),
            md5="851b7faac91e8e23940ee1294231d5c7",
        ),
    ),
    Model(
        name="Segment-Anything (accuracy)",
        encoder_weight=Weight(
            url=(
                "https://github.com/wkentaro/labelme/releases/download/sam-20230416/"
                "sam_vit_h_4b8939.quantized.encoder.onnx"
            ),
            md5="958b5710d25b198d765fb6b94798f49e",
        ),
        decoder_weight=Weight(
            url=(
                "https://github.com/wkentaro/labelme/releases/download/sam-20230416/"
                "sam_vit_h_4b8939.quantized.decoder.onnx"
            ),
            md5="a997a408347aa081b17a3ffff9f42a80",
        ),
    ),
]


def get_model(name: str = "Segment-Anything (Edge)") -> Model:
    for model in MODELS:
        if model.name == name:
            return model
    available = ", ".join(m.name for m in MODELS)
    raise ValueError(f"Unknown model '{name}'. Available: {available}")


def _cached_download(weight: Weight, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_kwargs = {"hash": f"md5:{weight.md5}"}
    if "hash" not in inspect.signature(gdown.cached_download).parameters:
        checksum_kwargs = {"md5": weight.md5}
    gdown.cached_download(weight.url, str(output_path), **checksum_kwargs)
    return output_path


def ensure_model_weights(
    model_name: str = "Segment-Anything (Edge)",
    models_dir: str | Path = Path(__file__).resolve().parent / "models",
) -> tuple[Path, Path]:
    model = get_model(model_name)
    target_dir = Path(models_dir)

    encoder_name = Path(model.encoder_weight.url).name
    decoder_name = Path(model.decoder_weight.url).name

    encoder_path = _cached_download(model.encoder_weight, target_dir / encoder_name)
    decoder_path = _cached_download(model.decoder_weight, target_dir / decoder_name)
    return encoder_path, decoder_path
