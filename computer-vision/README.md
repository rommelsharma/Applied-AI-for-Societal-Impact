# Computer Vision: Fundamentals & MLOps

An overview of computer vision (CV) foundations, common architectures, model ecosystems, and production MLOps workflows.

---

## What is Computer Vision?

Computer Vision is a specialized domain of AI that enables machines to understand and interpret images and video. It builds on:

- **Supervised / unsupervised / reinforcement learning**
- **CNNs and Vision Transformers** as core architectural families
- **Transfer learning and attention mechanisms** for practical efficiency
- **Multimodal representation learning** for cross-modal understanding

---

## Common CV Tasks

| Task | Examples |
|---|---|
| Classification | Scene recognition, species ID |
| Detection | Object localization, YOLO-style inference |
| Segmentation | Semantic, instance, panoptic |
| Other | OCR, anomaly detection, video understanding, visual search |

---

## Model Families & When to Use Them

| Model | Best for |
|---|---|
| ResNet / EfficientNet | General image classification |
| YOLO family | Real-time object detection |
| DETR / RT-DETR | Transformer-based detection |
| SAM | Segmentation & annotation acceleration |
| CLIP | Image–text similarity and retrieval |
| DINOv2 | Feature extraction and embeddings |
| MobileNet / EfficientNet Lite | Edge and mobile inference |
| TrOCR / PaddleOCR | Document intelligence |
| Florence / GPT-4o Vision | Multimodal reasoning |

---

## Key Libraries & Ecosystems

- **TIMM (PyTorch Image Models)** — hundreds of pretrained CNNs and ViTs, standardized APIs, strong Hugging Face integration.
  ```python
  import timm
  model = timm.create_model('vit_base_patch16_224', pretrained=True)
  ```
- **Hugging Face** — thousands of pretrained CV and multimodal models
- **AWS** — Rekognition, SageMaker JumpStart
- **Google Cloud** — Vertex AI Vision, Model Garden
- **NVIDIA** — TAO Toolkit, TensorRT, DeepStream

---

## MegaDetector

An object detection model widely used in wildlife camera trap workflows. Detects animals, humans, and vehicles as a pre-filter before downstream species classification — significantly reducing manual review workload. Supports edge deployment in remote environments and integrates with active learning pipelines.

---

## MLOps Pipeline (7 Layers)

| Layer | Key Tools |
|---|---|
| **Data** | DVC versioning, annotation store, feature store |
| **Experiment tracking** | MLflow, W&B, Optuna / Ray Tune |
| **Training** | Distributed GPU/k8s, augmentation, mAP/IoU eval |
| **CI/CD** | Pytest, canary/shadow splits, ONNX export |
| **Model registry** | Versioned artifacts, staged promotion, rollback |
| **Serving** | Triton, TorchServe, TensorRT, k8s HPA |
| **Monitoring** | Evidently, Alibi, Airflow/Kubeflow retraining loop |

> Data quality and annotation consistency are often more important than model complexity.

---

## Further Reading

- [TIMM Documentation](https://huggingface.co/docs/timm)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [MegaDetector GitHub](https://github.com/microsoft/CameraTraps)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [AWS SageMaker](https://docs.aws.amazon.com/sagemaker/)
- [Google Vertex AI Vision](https://cloud.google.com/vision)
- [NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit)
