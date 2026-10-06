# Initial source-delivery software verification

This records the earlier software-only checks. The later real PlantVillage dataset audit is in `reports/PLANTVILLAGE_STATUS.md`; no real-data training has occurred.

Verified on 2026-10-06, Linux x86_64, Python 3.12.14, with the project's pinned dependencies installed. This is a source-code deliverable, not a trained disease classifier.

## Executed checks

- **28 pytest tests passed.** The suite covers image labels, corrupt/zero-byte/unsupported/small images, grayscale handling, empty-data audit, editable metadata preservation, stale derived metadata rejection, exact/near duplicates and transitive clusters, label conflicts, group/hash separation, missing classes, fallback/small-test flags, dry-run non-mutation, locked-file tampering/additions, receipt invalidation, no test-image reads during training preflight, final-evaluation flag enforcement, exposure blocking, actual TensorFlow class order, preprocessing consistency, prediction serialization and confidence wording, missing ImageNet failure, and export split restrictions.
- An isolated end-to-end test used a **tiny test-only network and temporary random image fixtures** to exercise the two-stage training plumbing, validation reports, gated final evaluation, frozen artifact verification, annotated inference, both TFLite exports, and the post-exposure training block. Those fixtures are software test inputs, not tomato data. No fixture or fixture-derived model is included in `data/raw`, `models`, or the download as a usable disease classifier. Fixture metrics are deliberately not reported as disease performance.
- The **actual MobileNetV3Small architecture** successfully loaded ImageNet weights and constructed a three-output model. Parameter count: **1,013,363**. BatchNorm layers remained frozen during the fine-tuning setup check.
- The actual architecture's inference graph converted successfully to float32 TFLite (**4,029,632 bytes**) and dynamic-range TFLite (**1,185,688 bytes**) in a temporary directory. A single random converter-input fixture produced maximum absolute probability differences of about **0.00000054** and **0.01003**, respectively. These numbers are only converter sanity checks for an untrained classification head, not validation consistency or accuracy evidence for a trained disease classifier.
- Python source compilation succeeded. Source code was formatted for readability.

Command used for the suite:

```bash
TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests
```

The environment emitted 14 third-party Matplotlib/Pyparsing deprecation warnings, with no test failures. They do not represent accuracy warnings or data leakage.

## Not established

No real dataset was provided. Therefore no real-data duplicate audit, real source-independence verification, dataset split, disease-model training, model accuracy, independent holdout performance, field/drone validity, or Raspberry Pi hardware benchmark is claimed. The production pipeline intentionally refuses to train from the empty supplied dataset.

Windows setup commands are supplied but were not executed on Windows here. Offline/ARM installation and device-specific TFLite runtimes were not exercised. Resume restores optimizer/epoch/best checkpoint but restarts callback patience and shuffle state; exact uninterrupted-training equivalence is not claimed.

The checksum/receipt design protects against accidental changes, not malicious modification by someone who controls the project files. Duplicates and scientifically invalid grouping can still escape automated detection; read the README limitations and review the data.
