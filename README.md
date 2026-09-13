# Single Object Detection with MobileNetV2

A from-scratch single object detector built on a MobileNetV2 backbone: given an image, the model predicts whether an object is present and regresses its bounding box (`x, y, w, h`). Built as part of the **rubythalib.ai** Computer Vision bootcamp track (mentor: Daniel Syahputra).

Unlike multi-object detectors (YOLO, SSD, Faster-RCNN), this model assumes **one object per image** — a deliberately simpler design used here as a teaching vehicle for the core building blocks of object detection: a shared backbone, a joint classification + regression head, and a mixed loss function.

📁 **Dataset**: [Google Drive](https://drive.google.com/drive/folders/19ti0TK39f7SYJ06YQuxnXoLvVrp7xNhK?usp=sharing) (241 images)

---

## Results

Three iterations were trained, each changing one hypothesis about what was limiting bounding-box accuracy. All numbers below are from real training runs (not estimated) — see `notebooks/training.ipynb` and `experiments/training_v2.ipynb` for the executed notebooks with full epoch logs.

| Version | Best IoU (test) | IoU (train, final epoch) | Train/test gap | Key change |
|---|---|---|---|---|
| v1 (baseline) | 0.7186 | 0.8382 | 0.120 | Full fine-tune, single combined head, plain MSE loss, no augmentation |
| v2 | 0.7276 | 0.8085 | 0.081 | + Color-jitter augmentation (train split only) + partial backbone freeze (14/19 MobileNetV2 blocks frozen) |
| v3 (current) | **0.8100** | 0.8339 | **0.024** | + Split prediction head (bbox branch keeps a 4×4 spatial grid instead of collapsing to 1×1) + differentiable IoU loss added alongside MSE |

v3 is the best-performing version on the held-out test set, both in raw IoU and in train/test gap (least overfit). It is the version used in `notebooks/training.ipynb` and the serving code in this repo.

Dataset split: 241 images → 192 train / 49 test (80/20). v1 trained on a Tesla T4 GPU; v2 and v3 were trained on CPU (Colab GPU quota was unavailable at the time) — still practical given MobileNetV2's small size and the dataset's size.

### A note on v1's notebook

v1 was iterated on directly into v2 (adding augmentation + partial freeze on top of the same base script), so a standalone v1 notebook file was not preserved separately — its configuration and results are documented above from the original run's logs. `experiments/training_v2.ipynb` is the earliest fully-preserved notebook.

---

## Key finding: possible "mode-based" localization rather than true per-image localization

Beyond the aggregate IoU numbers, testing the served model on real-world photos **outside** the training distribution (different cars, countries, lighting — sourced from the web, not the original dataset) surfaced a more interesting pattern than "sometimes accurate, sometimes not":

- On two different bright/clear-sky test images (different cars, different positions in frame), v3 produced bounding boxes that differed by **less than 1%** in every coordinate.
- On a dark/overcast test image, v3 produced a noticeably different (about 1.45× larger) box — so the output is not a literal constant.

This pattern is consistent with a hypothesis worth flagging rather than a proven conclusion (only a handful of out-of-distribution images were tested, not a systematic study): the model may be picking up on **coarse global image statistics** (overall brightness/contrast) and outputting one of a few learned "typical" box shapes for that lighting regime, rather than tightly localizing each object's actual edges. This would explain most of what we observed across all three versions — loose boxes in dramatic lighting, a box that fully failed on a busy multi-car scene, and why "fixing" one qualitative failure (v2's box clipping the car) via v3 immediately introduced a different one (v3 being the loosest box of all three on that same image) rather than converging to a clean fix.

See `results/gallery.jpg` for the full set of qualitative test images referenced above, and `results/*.jpg` for individual annotated examples.

### Known limitations

- **241 images is a small dataset.** All three versions show a real (if shrinking) train/test IoU gap.
- **Single-object assumption breaks on busy scenes.** Tested on a highway photo with 8+ visible cars — the model produced one oversized box spanning several vehicles rather than picking one (see `results/v1_multi_objek_gagal.jpg`). This is expected: the architecture has no mechanism to choose among multiple candidate objects.
- **Sensitivity to lighting/scene conditions**, plausibly tied to the "mode-based localization" pattern above rather than genuine edge-precise localization.
- No systematic held-out benchmark exists for the out-of-distribution test images (they were used for qualitative debugging, not for a rigorous accuracy claim) — treat the "Key finding" above as a hypothesis for further investigation, not a settled result.

---

## Architecture

```
Input Image (224×224×3)
        │
        ▼
  MobileNetV2 backbone (ImageNet-pretrained; blocks 0-13 frozen, 14-18 fine-tuned)
        │
        ├──► objectness_head: AdaptiveAvgPool2d(1×1) → FC(1280→256) → FC(256→1) → sigmoid → is_object
        │
        └──► bbox_head: Conv1×1(1280→256) → AdaptiveAvgPool2d(4×4) → FC(4096→512) → FC(512→4) → sigmoid → (x, y, w, h)
```

**Loss** (`MixedLoss`): `BCE(is_object) + MSE(bbox) + IoU_loss(bbox)`, the IoU term computed only on samples that actually contain an object.

---

## Repository structure

```
notebooks/
  training.ipynb                     Main training notebook (v3 — current architecture), run on Colab
  model_serving_colab_ngrok.ipynb    Serving via FastAPI + ngrok tunnel, with bbox visualization (Colab)
experiments/
  training_v2.ipynb                  Earlier iteration (augmentation + partial freeze), kept for reference
serving/
  app.py                             Local FastAPI serving script (no Colab/ngrok dependency)
results/
  gallery.jpg                        Side-by-side comparison across lighting/scene conditions and versions
  v1_*.jpg, v3_*.jpg                 Individual annotated test examples
requirements.txt
LICENSE
```

---

## Setup & usage

### 1. Training

1. Open `notebooks/training.ipynb` in Google Colab.
2. Download the [dataset](https://drive.google.com/drive/folders/19ti0TK39f7SYJ06YQuxnXoLvVrp7xNhK?usp=sharing) into your own Google Drive, and update `DATASET_FOLDER` in the notebook to match its path.
3. Run all cells. The notebook automatically backs up `best.pt`/`last.pt` to `MyDrive/single_object_detection/checkpoints/` at the end of training.

### 2. Serving

**Local:**
```bash
pip install -r requirements.txt
# place best.pt in the same directory as app.py
cd serving
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Colab (with public URL via ngrok):**
Open `notebooks/model_serving_colab_ngrok.ipynb`, replace the ngrok auth token placeholder with your own (get one free at [dashboard.ngrok.com](https://dashboard.ngrok.com)), and run all cells.

Both expose `POST /predict` (multipart file upload) → JSON response:
```json
{
  "is_object": true,
  "bbox": {"x": 546.7, "y": 324.1, "w": 664.3, "h": 323.5}
}
```

⚠️ **Architecture compatibility**: the serving code's `ObjectDetectionModel` class must match whichever checkpoint you load. The code in this repo matches **v3**. Loading a v1/v2 checkpoint into it will fail with a `Missing key(s)/Unexpected key(s)` error — this happened once during development (see commit history) and is worth knowing about before swapping checkpoints.

---

## License

MIT — see [LICENSE](LICENSE).
