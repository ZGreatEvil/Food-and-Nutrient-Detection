# Food and Nutrient Detection

A deep learning application that detects food items in images and estimates their nutritional content. Built on a custom-trained EfficientNetB0 model with a sliding window detection approach, served through a Gradio web interface.

---

## Table of Contents

- [About](#about)
- [How It Works](#how-it-works)
- [Nutritional Output](#nutritional-output)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)

---

## About

Upload a food image and the system identifies food items present using a trained EfficientNetB0 classifier, then looks up and aggregates nutritional values for all detected items. Detection uses a multi-scale sliding window approach with IoU-based deduplication to handle multiple foods in a single image.

---

## How It Works

**1. Model**
EfficientNetB0 trained on the NutritionVerse dataset (COCO format annotations). The model file is `EfficientNetB0_NoTrain_E64_B128_IMG224.h5` — trained for 64 epochs, batch size 128, input resolution 224×224.

**2. Sliding Window Detection**
The input image is scanned at three window scales (1/2, 1/3, 1/4 of image dimensions) with multiple step sizes. Each window is resized to 224×224, preprocessed, and batched for inference.

**3. Filtering**
Detections below 0.9 confidence are discarded. Remaining boxes are deduplicated using IoU — overlapping boxes of the same class with IoU > 0.1 are suppressed, keeping only the highest-confidence detection.

**4. Nutrition Lookup**
Detected food categories are matched against `food_nutrients.csv` and summed to produce a total nutritional breakdown for the full image.

---

## Nutritional Output

For each detected food, the following nutrients are reported and totalled:

- Calories (kcal)
- Fat, Carbohydrates, Protein (g)
- Calcium, Iron, Magnesium, Potassium, Sodium (g)
- Vitamin D, Vitamin B12 (g)

---

## Dataset

- **Images & Annotations**: [NutritionVerse](https://github.com/JenWike/nutritionverse) — COCO-format annotated food images (`nutritionverse-data/`)
- **Nutrition Data**: `food_nutrients.csv` — per-food nutritional values
- **Dish Metadata**: `nutritionverse_dish_metadata3.csv` — processed dish metadata

---

## Project Structure

```
Food-and-Nutrient-Detection/
├── app.py                                          # Main Gradio application
├── food_calorie_prediction.ipynb                   # Model training notebook
├── food_metatdata_converter.ipynb                  # Metadata preprocessing
├── inference_food.ipynb                            # Inference testing notebook
├── gradio.ipynb                                    # Gradio prototyping notebook
├── food_nutrients.csv                              # Nutrition lookup table
├── nutritionverse_dish_metadata3.csv               # Processed dish metadata
├── nutritionverse-data/                            # NutritionVerse dataset
├── EfficientNetB0_NoTrain_E64_B128_IMG224.h5       # Trained model weights
├── EfficientNetB0_NoTrain_E64_B128_IMG224_...log   # Training log
├── requirements.txt                                # Python dependencies
└── .gradio/                                        # Gradio cache
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- GPU recommended for faster inference

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/ZGreatEvil/Food-and-Nutrient-Detection.git
cd Food-and-Nutrient-Detection
```

---

## Usage

```bash
python app.py
```

This launches a Gradio web interface. Upload any food image to receive:
- The original image with bounding boxes and confidence scores drawn on detected items
- A list of all detected food categories
- A full nutritional breakdown totalled across all detections

A public shareable link is also generated automatically via `share=True`.

---

## License

This project is licensed under the [MIT License](./LICENSE).
