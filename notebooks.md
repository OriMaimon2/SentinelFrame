# SentinelFrame – Initial Training Experiments

## Synthetic Dataset Generation

As an initial proof of concept, a small synthetic dataset was generated using diffusion models.

- **20 images were generated for each label**
- **Total generated images:** 160
- **Generation cost:** approximately **₪20**

This initial dataset was used to validate the complete training pipeline before scaling to a larger dataset.

---

# Training Pipeline

Two Google Colab notebooks were developed:

## 1. Data Preprocessing Notebook

This notebook performs the preprocessing pipeline for the raw generated images.

At this stage, **no data augmentation is applied**. The notebook only performs preprocessing and dataset preparation.

---

## 2. Training Notebook

This notebook performs the complete model training procedure, including:

- Loading the processed dataset
- Training the classifier
- Validation
- Model checkpointing
- Final evaluation on the test set

---

# Running the Project

Both notebooks were uploaded to Google Colab.

The notebooks are located in this directory and are ready to run.

The file paths are already configured for Google Colab.

To reproduce the experiments:

### Preprocessing Notebook

Upload:

- the preprocessing notebook
- the `tar.gz` archive containing the generated synthetic images

### Training Notebook

Upload:

- the training notebook
- `generate_images.py`
- `config.py`
- any additional project files required by the notebook

---

# Experiment 1 – Original Synthetic Dataset

Dataset split:

- Total images: **160**
- Training: **112**
- Validation: **24**
- Test: **24**

Training configuration:

- **100 epochs**

Best checkpoint:

```
Loaded best checkpoint from epoch 20
(val_f1_macro = 0.4786)
```

### Test Results

| Metric | Value |
|---------|------:|
| Precision (Macro) | **0.6771** |
| Recall (Macro) | **0.5000** |
| F1 (Macro) | **0.5405** |
| mAP (Macro) | **0.7424** |
| Subset Accuracy | **0.4167** |
| Loss | **0.8835** |

---

# Experiment 2 – Data Augmentation

To increase dataset diversity, several augmentations were applied **only to the training set**, including:

- Gaussian noise
- Random rotations
- Additional image perturbations

An important design decision was to perform **all augmentations before the preprocessing pipeline**.

This means that every augmented image is treated as a new raw CCTV image, allowing the preprocessing pipeline to process both original and augmented images identically.

Dataset split:

- Total images: **496**
- Training: **448**
- Validation: **24**
- Test: **24**

Training configuration:

- **100 epochs**

Best checkpoint:

```
Loaded best checkpoint from epoch 5
(val_f1_macro = 0.5393)
```

### Test Results

| Metric | Value |
|---------|------:|
| Precision (Macro) | **0.3812** |
| Recall (Macro) | **0.5833** |
| F1 (Macro) | **0.4477** |
| mAP (Macro) | **0.6549** |
| Subset Accuracy | **0.2083** |
| Loss | **0.8948** |

Although the training set became significantly larger, this augmentation strategy alone did not improve generalization.

---

# Experiment 3 – Larger Synthetic Dataset + More Augmentations

Next, the number of generated synthetic images was doubled.

In addition, the augmentation pipeline was expanded to produce **five augmented versions** of each training image.

Dataset split:

- Total images: **496**
- Training: **448**
- Validation: **24**
- Test: **24**

Training configuration:

- **100 epochs**

Best checkpoint:

```
Loaded best checkpoint from epoch 16
(val_f1_macro = 0.8783)
```

### Test Results

| Metric | Value |
|---------|------:|
| Precision (Macro) | **0.8214** |
| Recall (Macro) | **0.7917** |
| F1 (Macro) | **0.8002** |
| mAP (Macro) | **0.8568** |
| Subset Accuracy | **0.7500** |
| Loss | **0.9869** |

These results demonstrate a substantial improvement over the previous experiments.

The combination of a larger synthetic dataset and increased augmentation diversity significantly improved the model's ability to generalize.

---

# Current Model

All experiments described above were performed using **OpenCLIP ViT-B/32** as the image backbone.

The results indicate that foundation vision models are highly effective for the SentinelFrame frame-level hand-state recognition task.

---

# Next Step

The next stage of the project is to evaluate additional backbone architectures under the same training protocol.

The selected models are:

- OpenCLIP ViT-B/32 (baseline)
- DINOv2
- MobileNetV3
- EfficientNetV2
- SigLIP
- ConvNeXt V2
- ResNet-18 (trained from scratch)

These experiments will compare both lightweight convolutional networks and modern foundation vision models in order to identify the most suitable backbone for the SentinelFrame framework.