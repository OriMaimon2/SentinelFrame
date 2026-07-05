# SentinelFrame – Frame-Level Hand State Recognition for Retail CCTV

## Project Overview

SentinelFrame is a computer vision system for analyzing individual people detected in supermarket CCTV footage.

The objective is **not** to detect theft or identify shoplifters. Instead, the system classifies **observable hand states and interactions** for each detected person in every frame. These low-level observations can later be aggregated by a separate temporal module or risk engine to detect suspicious behavioral patterns.

The project focuses on building a robust **frame-level multi-label classifier** trained primarily on synthetic data.

---

# Pipeline

The complete inference pipeline is:

```
Video Frame
      │
      ▼
YOLO Person Detection (pretrained)
      │
      ▼
Person Bounding Box
      │
      ▼
Expand Bounding Box (≈20%)
      │
      ▼
Crop Person Region
      │
      ▼
Letterbox Resize
      │
      ▼
224×224 RGB Image
      │
      ▼
Frame-Level Multi-Label Classifier
      │
      ▼
Per-Person Labels + Confidence Scores
```

---

# Input

The classifier receives a single cropped person image extracted from one CCTV frame.

Input preprocessing:

* Detect people using a pretrained YOLO model.
* Expand each bounding box by approximately 20% to preserve surrounding context (shopping bags, shelves, baskets, etc.).
* Crop the expanded region.
* Apply Letterbox resizing to a fixed resolution while preserving the original aspect ratio.
* Feed the resulting RGB image into the classifier.

The classifier operates **independently on every frame**.

---

# Output

The model performs **multi-label classification**.

Each label is predicted independently using a sigmoid activation.

The output is a confidence score between 0 and 1 for every label.

Example:

```python
{
    "hand_in_pocket": 0.91,
    "hand_in_bag": 0.04,
    "hand_under_clothing": 0.01,
    "object_in_hand": 0.73,
    "interacting_with_shelf": 0.68,
    "hand_occluded_generic": 0.05,
    "both_hands_not_visible": 0.00
}
```

---

# Label Space

The classifier predicts the following observable states:

```python
LABELS = [
    "hand_in_pocket",
    "hand_in_bag",
    "hand_under_clothing",
    "object_in_hand",
    "interacting_with_shelf",
    "hand_occluded_generic",
    "both_hands_not_visible"
]
```

## Label Definitions

### hand_in_pocket

At least one hand is clearly inside a pants, jacket, or coat pocket.

---

### hand_in_bag

At least one hand is inside a shopping bag, handbag, backpack, or other carried bag.

---

### hand_under_clothing

A hand is hidden underneath clothing, such as inside a jacket, hoodie, sweater, or shirt.

---

### object_in_hand

The person is visibly holding a product or object.

---

### interacting_with_shelf

The person is visibly reaching toward, touching, picking up, returning, or examining products on a supermarket shelf.

---

### hand_occluded_generic

A hand is not visible due to normal occlusion.

Examples include:

* hidden behind the person's body
* blocked by another customer
* occluded by shelves
* outside the camera field of view
* hidden by shopping carts or baskets

This label intentionally represents **non-suspicious visibility loss**.

---

### both_hands_not_visible

Neither hand is visible in the current frame.

This label is independent of the reason for the occlusion.

---

# Model Characteristics

* Frame-level classification only
* Multi-label classification
* Sigmoid output layer
* BCEWithLogitsLoss during training
* Independent prediction for each label
* No temporal information is used by the classifier

---

# Synthetic Dataset

The classifier will be trained primarily using synthetic images generated with diffusion models.

Each generated image must satisfy:

* realistic supermarket environment
* realistic CCTV perspective
* overhead or corner-mounted surveillance camera
* wide-angle lens
* realistic customer behavior
* high visual diversity
* realistic lighting
* compression artifacts
* sensor noise
* slight motion blur
* surveillance-camera appearance

Every label must contain at least **100 high-quality samples**, with balanced representation across the dataset.

Generated images should vary in:

* customer appearance
* clothing
* supermarket layouts
* shelf configurations
* shopping baskets
* shopping carts
* bags
* camera angles
* lighting conditions
* image quality
* crowd density

Augmentation will further improve diversity through:

* Gaussian noise
* motion blur
* JPEG compression
* brightness and contrast changes
* perspective distortion
* CCTV artifacts

---

# Future Extensions

The current project intentionally performs only frame-level reasoning.

Future modules may include:

* person tracking
* temporal aggregation
* risk scoring
* event detection
* suspicious behavior recognition
* real-time deployment

These components are intentionally separated from the classifier so that the classifier remains responsible only for estimating observable visual states.
