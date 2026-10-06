# Tflite Full Validation Comparison



```json
{
  "scope": "validation only; NOT independent test performance",
  "sample_count": 440,
  "class_order": [
    "early_blight",
    "healthy",
    "late_blight"
  ],
  "keras_model_sha256": "422bbd1a89e29c0c533ffe95e749baa8936d5f9f1dd2d2090fae234eafed0171",
  "results": {
    "keras": {
      "loss": 0.11638785153627396,
      "accuracy": 0.9568181818181818,
      "macro_precision": 0.9573209337039651,
      "macro_recall": 0.9577694235588973,
      "macro_f1": 0.9565985410261048,
      "weighted_precision": 0.9577398703173945,
      "weighted_recall": 0.9568181818181818,
      "weighted_f1": 0.9563121721830349,
      "sample_count": 440,
      "class_metrics": {
        "early_blight": {
          "precision": 0.9784172661870504,
          "recall": 0.8947368421052632,
          "f1-score": 0.9347079037800687,
          "support": 152.0
        },
        "healthy": {
          "precision": 0.9487179487179487,
          "recall": 1.0,
          "f1-score": 0.9736842105263158,
          "support": 148.0
        },
        "late_blight": {
          "precision": 0.9448275862068966,
          "recall": 0.9785714285714285,
          "f1-score": 0.9614035087719298,
          "support": 140.0
        }
      },
      "warnings": []
    },
    "float32": {
      "loss": 0.11638794094324112,
      "accuracy": 0.9568181818181818,
      "macro_precision": 0.9573209337039651,
      "macro_recall": 0.9577694235588973,
      "macro_f1": 0.9565985410261048,
      "weighted_precision": 0.9577398703173945,
      "weighted_recall": 0.9568181818181818,
      "weighted_f1": 0.9563121721830349,
      "sample_count": 440,
      "class_metrics": {
        "early_blight": {
          "precision": 0.9784172661870504,
          "recall": 0.8947368421052632,
          "f1-score": 0.9347079037800687,
          "support": 152.0
        },
        "healthy": {
          "precision": 0.9487179487179487,
          "recall": 1.0,
          "f1-score": 0.9736842105263158,
          "support": 148.0
        },
        "late_blight": {
          "precision": 0.9448275862068966,
          "recall": 0.9785714285714285,
          "f1-score": 0.9614035087719298,
          "support": 140.0
        }
      },
      "warnings": [],
      "top1_agreement_with_keras": 1.0,
      "changed_predictions": 0,
      "max_absolute_probability_difference": 7.510185241699219e-06
    },
    "dynamic_range": {
      "loss": 0.13335713744163513,
      "accuracy": 0.9477272727272728,
      "macro_precision": 0.9475024596787431,
      "macro_recall": 0.9482456140350877,
      "macro_f1": 0.9473979998071931,
      "weighted_precision": 0.9477325685342703,
      "weighted_recall": 0.9477272727272728,
      "weighted_f1": 0.9472426095837136,
      "sample_count": 440,
      "class_metrics": {
        "early_blight": {
          "precision": 0.951048951048951,
          "recall": 0.8947368421052632,
          "f1-score": 0.9220338983050848,
          "support": 152.0
        },
        "healthy": {
          "precision": 0.9548387096774194,
          "recall": 1.0,
          "f1-score": 0.976897689768977,
          "support": 148.0
        },
        "late_blight": {
          "precision": 0.9366197183098591,
          "recall": 0.95,
          "f1-score": 0.9432624113475178,
          "support": 140.0
        }
      },
      "warnings": [],
      "top1_agreement_with_keras": 0.9681818181818181,
      "changed_predictions": 14,
      "max_absolute_probability_difference": 0.6159762740135193
    }
  },
  "deployment_recommendation": "Use float32 TFLite for closest fidelity to the selected Keras model. Dynamic-range quantization changes predictions and probabilities; it is included for review, not recommended as a drop-in equivalent."
}
```
