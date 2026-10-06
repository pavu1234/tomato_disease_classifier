# Tflite Export Report



```json
{
  "architecture": "MobileNetV3Small",
  "class_order": [
    "early_blight",
    "healthy",
    "late_blight"
  ],
  "sample_count": 30,
  "sample_counts_by_class": {
    "early_blight": 10,
    "healthy": 10,
    "late_blight": 10
  },
  "sample_selection": "First up to 10 validation images per class in deterministic filename order",
  "comparison_split": "validation only",
  "required_preprocessing": "EXIF transpose, RGB, float32 [0,255], bilinear resize to model height/width. Normalization is embedded. Do not divide pixels by 255.",
  "input_shape": [
    null,
    224,
    224,
    3
  ],
  "deployment_note": "On Raspberry Pi use a compatible TensorFlow Lite interpreter, allocate tensors, feed NHWC float32, and ship class_names.json. Benchmark on the actual Pi; no hardware latency is claimed.",
  "exports": {
    "float32": {
      "path": "/workspace/scratch/18bae1faa97c/tomato_disease_classifier/models/tomato_disease_mobilenetv3_float32.tflite",
      "size_bytes": 4030360,
      "compression_vs_keras_percent": 55.66339620831591,
      "top1_agreement": 1.0,
      "max_absolute_probability_difference": 3.248453140258789e-06,
      "review_required": false,
      "note": "Agreement on a validation subset is not deployment accuracy; Keras archive may include optimizer state."
    },
    "dynamic_range": {
      "path": "/workspace/scratch/18bae1faa97c/tomato_disease_classifier/models/tomato_disease_mobilenetv3_dynamic_range.tflite",
      "size_bytes": 1186360,
      "compression_vs_keras_percent": 86.94926178448021,
      "top1_agreement": 0.9333333333333333,
      "max_absolute_probability_difference": 0.3779188394546509,
      "review_required": true,
      "note": "Agreement on a validation subset is not deployment accuracy; Keras archive may include optimizer state.",
      "compression_vs_float32_percent": 70.56441608193809
    }
  }
}
```
