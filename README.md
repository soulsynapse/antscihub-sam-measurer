# antscihub-sam-measurer

Lightweight desktop GUI to measure things using SAM.

## General Workflow

1. Run scale-bar config on the image that contains the scale bar.

   ```powershell
   python antscihub-sam-measurer/scale_bar_config.py --source-input "path\to\image-folder\scale-bar-image.jpg"
   ```

   Click the two ends of the scale bar, enter the known length and unit, then
   press `Accept`. This writes
   `<scale-bar-image-stem>.scale_bar_config.result.json` next to that image.

2. Optionally precompute SAM embeddings overnight.

   ```powershell
   python antscihub-sam-measurer/precompute_embeddings.py --source-folder "path\to\image-folder"
   ```

   This walks the folder and all subfolders, downloads the selected model if
   needed, skips fresh embedding caches, and writes the same
   `<image>.<hash>.sam_embedding.npz` files that the GUI uses.

3. Run the SAM mask GUI on the images you want to measure.

   ```powershell
   python antscihub-sam-measurer/sam_hover_mask_gui.py
   ```

   Open images from the same folder, click objects to create masks, and let the
   GUI autosave each image's `.sam_clicks.json` and `.sam_clicks.npz` files.

4. Export the folder measurements to CSV.

   ```powershell
   python antscihub-sam-measurer/folder-to-csv.py --source-folder "path\to\image-folder"
   ```

   This reads the scale-bar config plus all saved SAM click metadata in that
   folder and writes `sam_mask_measurements.csv` back into the folder.

## What it does

- Loads Segment Anything ONNX encoder/decoder.
- Loads an image.
- Supports positive prompts and point erasing:
  - Left click: seed (or reseed) the current object
  - Right click: remove point(s) or committed mask(s) near the cursor
  - Mouse hover: temporary positive point for live preview
- Shows a live mask overlay as you move the mouse.
- Persists annotations as:
  - compact binary mask archive: `<image>.sam_clicks.npz` (bit-packed masks for smaller files)
  - human-readable metadata: `<image>.sam_clicks.json` (seed `x/y`, area, bbox, stats)
- Provides a scale-bar calibration helper that saves a JSON calibration next to
  the reference image.

## Setup

From the workspace root:

```powershell
python -m venv antscihub-sam-measurer/.venv
antscihub-sam-measurer/.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r antscihub-sam-measurer/requirements.txt
```

Required packages include `onnxruntime`, `numpy`, `Pillow`, and `gdown`.

## Run SAM Mask GUI

From the workspace root:

```powershell
python antscihub-sam-measurer/sam_hover_mask_gui.py
```

When you pick a model, the app automatically downloads missing ONNX weights and
loads the SAM sessions for you.

ONNX weights are stored in `antscihub-sam-measurer/models/`.

## SAM Outputs

SAM annotation files are saved next to the image being annotated:

- `<image>.sam_clicks.npz`: compressed binary masks, stored as bit-packed mask arrays.
- `<image>.sam_clicks.json`: readable metadata with image size, model name, threshold,
  mask count, total mask area in pixels, and per-mask seed point, bbox, area, and
  logit stats.
- `<image>.<hash>.sam_embedding.npz`: cached SAM embedding for faster reopening.

Runner mode also writes its configured result JSON, typically named like:

- `<stem>.sam_masks_from_clicks.result.json`

That result JSON points to the annotation files and reports saved mask counts and
total mask area.

## Precompute Embeddings

To precompute embeddings recursively for a folder tree:

```powershell
python antscihub-sam-measurer/precompute_embeddings.py --source-folder "path\to\image-folder"
```

If no folder is provided, the script opens a folder picker. It uses
`Segment-Anything (accuracy)` by default, downloads missing ONNX weights, and
skips images whose embedding cache is already fresh. Use `--force` to recompute
everything.

Optional summary file:

```powershell
python antscihub-sam-measurer/precompute_embeddings.py --source-folder "path\to\image-folder" --summary-json "path\to\summary.json"
```

## Scale Bar Config

Launch the scale-bar helper with an image or image folder:

```powershell
python antscihub-sam-measurer/scale_bar_config.py --source-input "path\to\image.jpg"
```

In the GUI, click two points on the reference scale bar, enter the known length
and unit, then press `Accept`.

When run directly, the scale-bar config is saved next to the selected/reference
image:

- `<image-stem>.scale_bar_config.result.json`

The JSON includes:

- selected image path and image size
- `point_a` and `point_b`
- measured `pixel_distance`
- `known_length` and `length_unit`
- `pixels_per_unit` and `units_per_pixel`
- the folder/image scope the calibration applies to

Runner mode writes the same payload to the explicit `--output-result-json` or
`--output-scale-bar-config-json` path.

## Folder CSV Export

After annotating images and creating a scale-bar config, export mask areas for a
folder:

```powershell
python antscihub-sam-measurer/folder-to-csv.py --source-folder "path\to\image-folder"
```

If no folder is provided, the script opens a folder picker. It reads the newest
`*.scale_bar_config.result.json` in the selected folder and all
`*.sam_clicks.json` files in that folder, then writes:

- `sam_mask_measurements.csv`

The CSV keeps the measurement columns first: image name, click number, pixel
area, `units_per_pixel` from the scale-bar config, computed area, and computed
area unit. It also appends flattened metadata columns from the scale-bar config
JSON, the annotation JSON, and each per-click record.

## Controls

- `Left click`: seed/reseed current object
- `Right click`: remove nearby point(s) and/or committed mask(s)
- `n`: start a new object
- `c`: clear prompts

## License and Attribution

This project is licensed under CC BY-NC 4.0. It was heavily inspired by
Annolid, which is copyright (c) 2024 Computational Physiology Laboratory and is
also distributed under CC BY-NC 4.0.

I made this project because I could not figure out how to get Annolid working
for my workflow, but Annolid's methods were clear enough to guide this smaller,
focused SAM measurement tool.

See [LICENSE](LICENSE) for the full local attribution notice.
