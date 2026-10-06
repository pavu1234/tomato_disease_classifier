# Tomato leaf disease classifier

**2026-10-06 training update:** The user approved the conservative 2,902-image PlantVillage subset. See `RUN_STATUS.md` for this bundle’s completion status, `TRAINED_MODEL_GUIDE.md` for inference/restoration commands, and `reports/approved_subset/` for the actual experiment. The earlier full-dataset audit is preserved in `reports/PLANTVILLAGE_STATUS.md` as historical evidence. Bulk images are not included in this ZIP.

A student-friendly Python project for **early_blight**, **healthy**, and **late_blight**, built around reproducible data handling and honest evaluation. Default model: ImageNet-pretrained MobileNetV3Small; configurable fallback: MobileNetV2. Neither architecture is initialized randomly if pretrained weights are unavailable.

**Delivery status:** Training is complete. The download contains the selected Keras model, both TFLite exports, populated class order, logs, and measured validation reports. Validation accuracy is **95.68%** and macro F1 is **0.9566** on 440 images. The 424-image locked test set has not been evaluated. Use the float32 TFLite export for fidelity; the dynamic-range export changed 14 validation predictions. Start with `TRAINED_MODEL_GUIDE.md`. The general pipeline below is for understanding/reproducing experiments; do not rerun splitting or training over the completed model. Bulk images are excluded; use `scripts/restore_data.py` if you need them.

## 1. Workflow

Real raw images → editable manifest → read-only audit → source/duplicate-aware split → training and validation → frozen selected model → explicitly authorized locked-test evaluation → TFLite export → single-image inference.

All choices are made using training and validation data. Final evaluation is never run by training. The final evaluator records exposure before decoding any test image and blocks subsequent training or splitting in that experiment. Repeating final evaluation is allowed only for the same frozen model and lock.

## 2. Install

Use **Python 3.10, 3.11, or 3.12**; Python 3.11 is recommended. This project pins TensorFlow 2.16.2 and Keras 3.3.3 for a reproducible API baseline, not the latest releases. Python 3.13+ is not supported by these pins. Install in an isolated virtual environment. A first model build needs internet access to download ImageNet weights, or a previously populated Keras weights cache.

Windows PowerShell, from the extracted project folder:

```powershell
cd tomato_disease_classifier
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, activation is optional. Run the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/create_manifest.py
.\.venv\Scripts\python.exe -m pytest -q
```

Linux/macOS:

```bash
cd tomato_disease_classifier
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CPU execution is supported; NVIDIA GPU training with modern TensorFlow on Windows typically needs WSL2. On ARM platforms, package availability differs: train/export on a supported computer and install a platform-compatible TFLite runtime on the Pi. Never imply desktop timings are Pi timings.

## 3. Supply real labelled data

Place real images in these folders. Nested folders are allowed:

```text
data/raw/early_blight/
data/raw/healthy/
data/raw/late_blight/
```

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp` (case-insensitive). No data is downloaded automatically. Verify dataset permission, label quality, provenance and intended use yourself. Use independently collected representative field images if field generalization is the intended claim. Do not curate a test set to produce a desirable score.

The audit flags unsupported, corrupt, zero-byte, very small (<32 pixels in either dimension by default), non-RGB and animated images. Corrupt, tiny, animated and unsupported images are excluded from splitting. Grayscale and other supported modes are converted safely to RGB in **processed copies**. EXIF orientation is applied and all processed images are losslessly written as PNG, so WEBP works consistently with the TensorFlow loader. Raw files are never changed or deleted by the software. Transparency is discarded by RGB conversion; review such images for suitability before locking.

```bash
python scripts/create_manifest.py
```

## 4. Edit and verify source metadata

Open `data/metadata/image_manifest.csv` with a spreadsheet editor that preserves strings, or a text editor. Do not edit hashes, dimensions, IDs, labels or paths; regenerate derived columns from raw data. Manual metadata is preserved on regeneration when the relative path matches.

Editable columns: `source_group`, `source_type`, `plant_id`, `capture_session`, `capture_date`, `camera_or_source`, `notes`.

**Group-wise splitting is only scientifically meaningful if `source_group` represents a true independent source, such as a plant ID, farm plot, capture session, date/camera session, or original dataset source. File names alone may not be enough.**

Examples of verified groups: `farm_A_plant_017`, `farm_B_session_2026_07_12_phone1`. Use the SAME group across class folders when the same plant/session appears in both. Choose the broadest independence boundary needed: if multiple plants share a highly correlated session, group by session, not just by plant. Filling `plant_id` alone does not change splitting; explicitly set `source_group` correctly.

Nested folders infer `inferred:session_name`; images directly in class folders receive stable `fallback:<image_id>` groups. Neither is verified. After establishing provenance, replace these IDs with meaningful group IDs and set `source_type` to **`verified_group`**. This is your declaration of verified provenance, not a claim that software can prove it. Do not set this merely to get past an error. At least three independent connected groups containing each class are needed.

If provenance cannot be recovered, explicitly use `--allow-fallback-groups`. Results must then be described as **preliminary internal performance**, not plant-independent or field performance.

## 5. Audit and split

```bash
python scripts/audit_dataset.py
python scripts/split_dataset.py --dry-run
python scripts/split_dataset.py
python scripts/verify_locked_test.py
```

Inspect `reports/raw_dataset_audit.md`, `exact_duplicates.csv`, `near_duplicates.csv`, and `label_conflict_candidates.csv` before splitting. Audit rescans raw files read-only, including corrupt files; splitting rejects stale or edited derived manifest metadata.

Duplicate detection combines SHA-256 and 64-bit perceptual hash with configurable Hamming threshold **6**. Same-source links and all detected duplicate links form transitive connected components. A component always remains in one split, even across classes. Exact or near duplicate **cross-label conflicts block splitting**. There is no automatic conflict override, relabeling or deletion. Resolve them manually using provenance and expert review; if you need to exclude files, curate a separate copy of the dataset, preserve the original, and regenerate the manifest.

Same-label near duplicates are flagged for manual review and kept together. This prevents detected cross-split overlap but duplicates within a split still reduce effective sample size. pHash can miss crops, rotations, edits and re-photographs; it can also join unrelated images. Review suspicious clusters. It is not proof of universal duplicate absence.

The seeded splitter searches 4,000 group assignments using class counts only, targeting 70%/15%/15%. It never ranks image difficulty or model outcomes. Large groups may make targets infeasible. It refuses missing classes, overlapping source/hash/near-duplicate clusters, and test counts below the configured minimum (20 per class). A candidate-search failure is not proof no mathematical solution exists; examine group sizes or add independent sources without breaking the intended independence boundary. Exact/near comparisons are exhaustive O(N²); very large datasets require more time and memory for their reports.

Optional explicit flags:

```bash
python scripts/split_dataset.py --dry-run --allow-fallback-groups --allow-small-test-set
python scripts/split_dataset.py --allow-fallback-groups --allow-small-test-set --seed 42
python scripts/split_dataset.py --force-rebuild
```

Only use these when justified. `--allow-small-test-set` permits a minimum of one image per class in each split, not credible statistical precision. Counts below 20 per class are always marked preliminary. `--force-rebuild` permits replacement of processed copies before model selection/exposure, never raw files. A completed selected model or exposure marker blocks rebuilding.

Dry runs produce `split_manifest_dry_run.csv`, `split_summary_dry_run.json/.md`, and `split_leakage_check_dry_run.json/.md`; they never overwrite authoritative split records or copy images. A committed split produces the requested `split_manifest.csv`, `split_summary.json/.md`, and `split_leakage_check.json/.md`, plus the cryptographic lock and verification receipt. Source-group assignments, seed, class counts, target deviations and leakage checks are recorded.

## 6. What the test lock guarantees

`data/metadata/test_lock_manifest.json` includes timestamp, seed, test paths, processed and original SHA-256 hashes, class labels, source groups, counts, all split assignments, a split-CSV checksum and a manifest checksum.

`verify_locked_test.py` hashes all processed files, validates complete inventories and group/hash separation, and writes a verification receipt with test-file size, modification time and change time. It produces `reports/test_lock_verification.json/.md`.

Training preflight validates train/validation hashes against locked assignments and confirms the prior full-test verification receipt still matches the lock and test-file metadata. **It never opens, hashes, decodes or loads test images.** Filesystem names/stat metadata are inspected solely for integrity. This reconciles test-lock validation with no test-image access during training. Run full verification again if metadata changes or the receipt is missing. A new split performs the initial full verification automatically.

Full cryptographic verification is repeated before final evaluation. This catches accidental modifications; it is not authenticated storage, OS access control or protection against an owner deliberately replacing locks/receipts/code. Use filesystem permissions and a separate data custodian for stronger isolation. Simultaneous editing/training/evaluation is unsupported: use one writer per experiment. Keep a backed-up immutable experiment copy. There is no globally enforceable exposure registry across project copies.

## 7. Train on train + validation only

```bash
python scripts/train_model.py
```

Optional:

```bash
python scripts/train_model.py --stage stage1
python scripts/train_model.py --skip-fine-tune
python scripts/train_model.py --stage1-epochs 20 --fine-tune-epochs 10 --batch-size 16
python scripts/train_model.py --resume
```

For resuming an interrupted run, repeat the same CLI overrides and stage flags. Resume restores the last complete epoch, optimizer, stage and best-checkpoint loss. Early-stopping/reduce-LR patience and shuffle streams restart: resumption is **not bitwise equivalent** to uninterrupted training. Partial epochs are rerun. A completed run refuses `--resume`.

Model: 224×224 RGB → training-only flip/rotation/zoom/contrast/brightness → pretrained MobileNetV3Small → global average pooling → dropout .30 → Dense 128 ReLU → dropout .20 → three-class softmax. MobileNetV3 normalization is embedded in its base; MobileNetV2 gets an explicit equivalent `[-1,1]` rescaling layer. All external inputs stay float32 RGB **[0,255]**, not [0,1]. Model inference disables augmentation and dropout. Resize is bilinear without aspect-ratio padding, consistently for every path.

Stage 1 freezes the base and trains the head, up to 30 epochs at Adam LR .001. Stage 2 reloads the best stage-1 checkpoint, unfreezes the last 25 base layers excluding BatchNorm, and trains up to 15 epochs at LR .00001. BatchNorm always runs with inference statistics. Both stages select checkpoints by validation loss. Final stage selection takes lower validation loss; macro F1 breaks exact loss ties. A worse fine-tuned stage never replaces the better head-only model.

Callbacks: best-only checkpoints, early stopping (patience 7, restore best), reduce-on-plateau (factor .3, patience 3, min LR 1e-6), CSV logger, NaN termination and an epoch-resume checkpoint. Class weighting uses **training labels only** when max/min class count >1.2. Shuffle, caching (off by default), deterministic seed, `tf.data` prefetch and actual saved loader class order are implemented. GPU/OS/library differences can still affect reproducibility.

Outputs include stage checkpoints, selected `best_model.keras`, last resume checkpoint, class order, training state/config/history, class-weight record, validation metrics, training curves and summary. A `frozen_model.json` ties the model, class order and preprocessing to the split. No locked-test metric is used in training or selection. Initial CLI epoch/batch overrides are recorded and do not require matching CLI options for later evaluation.

For another development experiment before final test exposure, use a separate experiment directory and a carefully tracked copy of the **same** locked split. Track all validation experiments and the original provenance. Frequent validation selection can overfit validation; no code can eliminate that statistical effect.

## 8. Validation evaluation

```bash
python scripts/evaluate_model.py --split val
```

Saves classification report, metrics, raw/normalized confusion matrices. Warnings flag accuracy <.75, macro F1 <.70, any class recall <.60, >80% predictions in one class, or <20 images in any class. Validation results are development evidence, not independent final performance. A frozen-model checksum and image/model-config consistency check prevent silently evaluating a modified artifact.

## 9. Final evaluation: only after all decisions are fixed

```bash
python scripts/evaluate_model.py --split test_locked --final-evaluation
```

Without `--final-evaluation`, access is refused before data loading. The command prints a warning, validates the lock, checks the frozen model and records exposure before reading pixels. It saves final metrics, class-wise/weighted/macro measures, confusion matrices, prediction CSV, and an honest summary. Timing is batched forward-pass wall time per image, including initial tracing, excluding decode: it is not Pi latency.

Once these results are seen, do not tune architecture, augmentation, hyperparameters, thresholds, preprocessing or source selection and then reuse the same holdout as a final independent test. A new experiment needs **genuinely unseen independent test sources**, not a reshuffle of exposed data. The software cannot prevent someone copying an experiment or viewing images outside the program; the scientific process remains your responsibility.

No confidence intervals are invented. Internal scores do not demonstrate field/drone performance. Small/unverified splits are labeled preliminary. No independent field dataset loader or field-validity claim is included.

Reporting template (replace brackets only with measured facts):

“On a locked internal test set containing [N] images across three classes, the model achieved [accuracy]% accuracy and [macro_f1] macro F1-score. The test split was separated using [grouping method]. These results estimate internal generalization only and do not by themselves demonstrate field or drone deployment performance.”

## 10. Export TensorFlow Lite

```bash
python scripts/export_model.py
```

Creates `models/tomato_disease_mobilenetv3_float32.tflite` and `models/tomato_disease_mobilenetv3_dynamic_range.tflite`. MobileNetV2 uses `mobilenetv2` filenames. Export strips training-only augmentation/dropout and verifies the serving graph matches Keras before conversion. The inference-only SavedModel conversion route is used. Dynamic-range quantization requires no representative calibration set. No full-int8 export is implemented.

Compares up to 10 deterministic validation images per class against Keras; reports file sizes, compression, top-1 agreement, maximum probability difference, preprocessing and Pi instructions. Divergence is marked for review, not hidden. These checks do not measure generalization. If you plan to choose an export format based on validation comparisons, do that **before** final evaluation. The requested command order below can export afterward when conversion is only packaging the already frozen model; final metrics remain Keras metrics, not TFLite accuracy.

On a Pi: copy the selected `.tflite`, `class_names.json` and preprocessing instructions; use a compatible TFLite interpreter, call `allocate_tensors()`, feed `[1,224,224,3]` float32 RGB [0,255], invoke, then map argmax using the saved class names. Use EXIF transpose and the same bilinear resize. OpenCV images are BGR and must be converted to RGB. Embedded normalization means **do not divide by 255**. Benchmark RAM, latency, thermals and accuracy on actual hardware and representative data before deployment.

## 11. Predict one image

```bash
python scripts/predict_image.py --image path/to/image.jpg
python scripts/predict_image.py --image path/to/image.jpg --save-annotated
```

JSON outputs go to `logs/predictions/`, optionally with an annotated PNG. Output includes class, confidence, per-class probabilities, inference time, model path, timestamp, scope note and low-confidence warning at the frozen .50 threshold. It verifies model/class-order hashes and output dimensions. The threshold is a warning only, never a fourth class.

“Confidence indicates classifier certainty for this image; it does not measure infected leaf area or disease severity.”

This closed-set classifier will choose one of three labels even for non-leaf images or other diseases. Softmax confidence is not calibrated diagnostic certainty. Non-tomato detection, disease severity, segmentation, pesticide choice and spray actuation are outside scope.

## 12. Exact requested command sequence

After real data and verified source metadata are supplied, run these **from the project folder**:

```bash
python scripts/create_manifest.py
python scripts/audit_dataset.py
python scripts/split_dataset.py --dry-run
python scripts/split_dataset.py
python scripts/verify_locked_test.py
python scripts/train_model.py
python scripts/evaluate_model.py --split val
python scripts/evaluate_model.py --split test_locked --final-evaluation
python scripts/export_model.py
pytest -q
```

Stop and inspect errors/reports before continuing. Do not bypass source/size/conflict blockers simply to obtain a score. For this trained release, use the approved configuration and saved-data restoration instructions. Do not rerun the general sequence over the completed model.

| Command or operation | Data used |
|---|---|
| Manifest, audit, split | Raw images and source metadata before holdout creation; no model outcomes |
| Full lock verification | Checksums of all processed splits; no decoding, training or metric computation |
| Class weights / optimizer updates | **Train only** |
| `train_model.py` | **Train + validation**; test lock/receipt metadata only |
| `evaluate_model.py --split val` | **Validation only** for predictions; train/val checksums + test metadata preflight |
| `evaluate_model.py --split test_locked --final-evaluation` | **Locked test only** for predictions/metrics; integrity hashes of all splits |
| `export_model.py` | **Validation only** for prediction consistency; no test images or calibration |
| `predict_image.py` | Only the explicitly provided image |
| `pytest -q` | Temporary generated software-test fixtures; never your raw or locked dataset |

## 13. Interpret metrics

- **Accuracy:** fraction of correctly classified images; can conceal poor minority-class performance.
- **Macro F1:** equal-weight mean of each class's precision/recall harmonic mean; useful for treating all three classes equally.
- **Weighted F1:** class F1 weighted by class support; can favor a large class.
- **Recall:** fraction of actual examples of a class found correctly. Low late-blight recall means many late-blight samples are missed.
- **Precision:** fraction of predictions of a class that are correct.
- **Confusion matrix:** rows are true labels, columns predicted labels. Row normalization shows per-class error rates.
- **Loss:** mean sparse categorical cross-entropy. Validation loss drives selection; low loss is not proof of field validity.

Always report split method, counts, source independence, duplicate handling, label verification and uncertainty/limitations alongside scores. 100% on duplicated or easy data is not the goal; credible lower performance on unseen representative data is more useful.

## 14. Limitations and troubleshooting

- Curated backgrounds, acquisition devices and disease stages differ from real field/drone scenes. Blurred canopy imagery may require a detection/segmentation task first.
- Source groups require truthful, meaningful metadata. Fallback groups do not represent known independent plants.
- Labels require expert verification; near-duplicate detection is approximate; correlated within-group images reduce effective sample size.
- Real agricultural deployment needs agronomist, regulatory and safety validation. Pesticide/spray actuation is outside scope.
- **No valid images:** put real images in all three class folders and regenerate manifest; the project does not fabricate samples.
- **ImageHash missing:** `pip install ImageHash Pillow`.
- **ImageNet download fails:** restore network access or correctly populate the standard Keras cache; training stops, with no random-weight fallback.
- **No feasible split:** examine class/group counts and connected clusters. Add independent groups; do not split a plant/session across partitions to satisfy ratios.
- **Unverified metadata:** verify provenance and replace inferred/fallback group IDs, or explicitly accept the preliminary limitation with the documented flag.
- **Hash/receipt failure:** stop; investigate which file changed. Do not recreate a lock around altered data to hide contamination. After a legitimate copy that only changed file timestamps, run full verification.
- **Out of memory:** before training, lower batch size using config or CLI; leave cache off. Model resolution is a development decision and is frozen before final evaluation.
- **TFLite conversion issues:** use the pinned dependencies in a clean environment. Check the report and interpreter compatibility on the Pi; do not silently switch preprocessing.
- **PowerShell activation blocked:** use `.\.venv\Scripts\python.exe` directly, as above.
- **Tests on a resource-constrained host:** set `TF_NUM_INTEROP_THREADS=1` and `TF_NUM_INTRAOP_THREADS=1` in your shell before `python -m pytest -q`.

## 15. Git and generated artifacts

`.gitignore` excludes raw/processed data, model binaries, test-lock/receipt/exposure files, logs, plots and generated CSV/JSON reports. Empty folders are supplied with `.gitkeep`. Because entire raw/processed folders are ignored, a new Git repository will not preserve those placeholders unless force-added; keep them locally or recreate the directories. Keep sensitive manifests/reports out of public repositories. Back up your entire experiment privately, including locks and source metadata; Git ignore is not a backup or a security boundary. Tests never populate delivered `data/raw`.

## 16. Reference documentation

- [TensorFlow MobileNetV3Small API and preprocessing contract](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small)
- [TensorFlow TFLiteConverter API](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter)

`FILE_INDEX.md` lists every supplied file and the complete folder tree. `VERIFICATION.md` records what was actually tested and what was not.
