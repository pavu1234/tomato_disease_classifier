# Training completed — PlantVillage approved subset

The selected MobileNetV3Small model is trained and frozen. Use `TRAINED_MODEL_GUIDE.md` to predict new images. No retraining or dataset download is needed for single-image inference.

| Measurement | Result |
|---|---:|
| Training images | 2,038 |
| Validation images | 440 |
| Locked test images, not evaluated | 424 |
| Validation accuracy | 95.68% |
| Validation macro F1 | 0.9566 |
| Validation cross-entropy loss | 0.116388 |
| Completed head-training epochs | 15 |
| Completed fine-tuning epochs | 15 |
| Selected fine-tuning epoch (1-based) | 12 |
| Completed training run, seconds | 356.6 |

Stage 1 stopped early after 15 epochs under the original 30-epoch limit. Fine-tuning completed its 15-epoch limit. The fine-tuned checkpoint was selected because its validation loss (0.116388) was lower than the head-only checkpoint (0.122649). No test results influenced this choice.

## Per-class validation metrics

| Class | Precision | Recall | F1 | Images |
|---|---:|---:|---:|---:|
| early_blight | 0.9784 | 0.8947 | 0.9347 | 152 |
| healthy | 0.9487 | 1.0000 | 0.9737 | 148 |
| late_blight | 0.9448 | 0.9786 | 0.9614 | 140 |

Early-blight recall is lower than the other classes; the validation confusion matrix records 16 missed early-blight images. A high overall score must not conceal this limitation.

## TFLite verification on all 440 validation images

| Export | Size (bytes) | Validation accuracy | Predictions changed vs Keras |
|---|---:|---:|---:|
| float32 | 4,030,360 | 95.68% | 0 |
| dynamic_range | 1,186,360 | 94.77% | 14 |

**Recommended export: float32.** It matched Keras's predicted class on all 440 validation images. Dynamic-range quantization changed 14 predictions and reduced validation accuracy from 95.68% to 94.77%; its probability differences can be substantial. The smaller file is included for review, not as a drop-in equivalent. These are validation checks, not independent deployment or test accuracy.

## Scientific scope and recovery history

The original import had 4,500 images. The user-approved conservative policy retained 2,902 images from 726 upstream leaf IDs and excluded components touching missing leaf metadata or cross-label similarity candidates. Original source images and labels were preserved. These results concern the curated, metadata-resolved PlantVillage subset, not all original images or realistic field/drone conditions. Leaf grouping does not establish plant/farm/session independence. Detected duplicate overlap across splits was zero, but pHash cannot prove all duplicates were found.

**The 424-image locked test set has not been evaluated. No final test accuracy is claimed.** Final evaluation requires the explicit flag after all decisions are fixed. Validation data was used for model selection, so its score is development performance, not an independent final estimate.

An earlier interrupted runtime lost its unsaved checkpoints. The successful run restarted with the same approved image list, seed, settings and split algorithm; no architecture/hyperparameter change was made based on the interim results. The successful run completed without resume. Durable recovery bundles were saved during this run. The recorded duration covers this successful training invocation, not download/recovery/export time.

The 28 software tests passed. Both exported files were invoked successfully, and the real selected Keras model passed a single-image/annotation smoke check using a validation image. No Raspberry Pi hardware benchmark or field validity is claimed.

## Package contents

Trained Keras checkpoints, float32 and dynamic-range TFLite files, class order, frozen model hashes, configuration, source code, exact split/provenance manifests, training logs/curves, validation reports, and restoration/prediction commands. Bulk images are excluded to keep the download small; `scripts/restore_data.py` replays the saved assignments from the pinned upstream source.
