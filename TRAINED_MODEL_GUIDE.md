# Using the trained PlantVillage model

Check `RUN_STATUS.md` first. Recovery bundles saved during training are not final releases. A completed release contains `models/frozen_model.json`, `reports/approved_subset/training_summary.json`, and both TFLite files.

## Predict an image on Windows

From the extracted `tomato_disease_classifier` directory, install the pinned requirements with Python 3.10–3.12 (3.11 recommended):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/predict_image.py --config config/plantvillage_approved.yaml --image "C:\path\to\leaf.jpg" --save-annotated
```

You do not need to retrain or download PlantVillage to predict a new image. The JSON and optional annotated image are written under `logs/predictions/`.

The model selects exactly one of `early_blight`, `healthy`, and `late_blight`. It cannot reliably reject other diseases or non-tomato images. Confidence measures classifier certainty, not disease severity or infected leaf area.

## Training population and scope

The approved subset contains 2,902 images from 726 upstream leaf IDs. The 1,598 excluded images belong to connected groups touching missing leaf metadata or cross-label near-duplicate candidates. Labels and source images were not changed.

| Class | Train | Validation | Locked test |
|---|---:|---:|---:|
| early_blight | 691 | 152 | 148 |
| healthy | 699 | 148 | 148 |
| late_blight | 648 | 140 | 128 |
| Total | 2,038 | 440 | 424 |

No source group or detected exact/near-duplicate pair crosses splits. Source groups represent upstream **leaf identities**, not verified independent plants, farms, or capture sessions. This curated subset does not establish field/drone performance.

Training uses seed 42, the original MobileNetV3Small/ImageNet model, up to 30 head-training epochs and 15 fine-tuning epochs, and validation-loss early stopping. Weights are updated only using training images. Validation data controls checkpoints/model selection. Test predictions are not produced by training.

**Export recommendation:** Use float32 TFLite. Across all 440 validation images it matched Keras predictions; dynamic-range quantization changed 14 predictions and reduced validation accuracy to 94.77%. See `RUN_STATUS.md`.

## Model files

- `models/best_model.keras`: selected trained Keras model, including preprocessing.
- `models/class_names.json`: actual class index order.
- `models/frozen_model.json`: model/class-order hashes and experiment configuration.
- `models/stage1_best_model.keras`: best head-training checkpoint.
- `models/last_model.keras`: last completed training epoch, for interrupted-run recovery.
- `models/tomato_disease_mobilenetv3_float32.tflite`: float32 export.
- `models/tomato_disease_mobilenetv3_dynamic_range.tflite`: smaller dynamic-range export.

TFLite inputs are NHWC float32 RGB, 224×224, values **0–255**. Apply EXIF orientation and bilinear resize. Normalization is inside the model: do not divide by 255. OpenCV reads BGR, so convert BGR to RGB. Ship the saved class-name list with the model. Export comparisons use up to 10 deterministic validation images per class; they are conversion checks, not final performance estimates or Pi benchmarks.

## Restore dataset files for reproducibility or final evaluation

The download excludes bulk training/test photos. It includes exact filenames, provenance, hashes and locked assignments. Restore from the pinned repository commit (Git must be installed):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git ../PlantVillage-source
git -C ../PlantVillage-source fetch origin 7f7ecc7e1eaca78107e3affe7cb5abd9427e139a
git -C ../PlantVillage-source checkout 7f7ecc7e1eaca78107e3affe7cb5abd9427e139a
git -C ../PlantVillage-source sparse-checkout set raw/color/Tomato___Early_blight raw/color/Tomato___Late_blight raw/color/Tomato___healthy leaf_grouping
python scripts/restore_data.py --source ../PlantVillage-source
```

The restoration script validates source hashes, replays saved assignments, recreates processed PNGs and verifies the original lock. It refuses mismatches rather than changing the lock. Use the pinned Pillow version; differences in the PNG encoder may otherwise cause a byte-check failure. No final test inference occurs during restoration.

After restoration, validation can be reproduced with:

```bash
python scripts/evaluate_model.py --config config/plantvillage_approved.yaml --split val
```

Only after all model/deployment choices are fixed, explicitly authorize final evaluation:

```bash
python scripts/evaluate_model.py --config config/plantvillage_approved.yaml --split test_locked --final-evaluation
```

This exposes the holdout and blocks further training in the experiment. Do not tune based on the results and report the same holdout as fresh. Validation figures in the delivered reports must not be presented as test accuracy.

For an **interrupted recovery bundle only**, restore the images, then resume with:

```bash
python scripts/train_model.py --config config/plantvillage_approved.yaml --resume
```

Resume restores the last epoch and optimizer but restarts patience/shuffle state. Completed models refuse resume. Do not regenerate splits for an existing model. See `README.md` for the general pipeline and `reports/approved_subset/` for this run's measured evidence.
