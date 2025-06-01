#!/usr/bin/env python
# coding: utf-8

# # Imports and Constants

# In[1]:


pip install gradio


# In[2]:


import gradio as gr
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import json
import os
import pandas as pd
from tensorflow.keras.applications.efficientnet import preprocess_input

# Constants
IMG_SIZE = 224  # Input size for the model


# # Load Model

# In[3]:


model_path = "EfficientNetB0_NoTrain_E64_B128_IMG224.h5"
model = tf.keras.models.load_model(model_path)


# # Load COCO Metadata

# In[4]:


base_path = 'nutritionverse-data'
coco_json_path = os.path.join(base_path, 'nutritionverse-manual/nutritionverse-manual/images/_annotations.coco.json')

with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)


# # Load Nutrition Data

# In[5]:


nutrition_data = pd.read_csv("food_nutrients.csv")


# # Prepare Label Encoder

# In[6]:


category_ids = [cat['id'] for cat in coco_data['categories']]
label_encoder = LabelEncoder()
label_encoder.fit(category_ids)


# # Nutrition Calculator

# In[7]:


def format_nutrition_info(nutrition_info):
    if not nutrition_info:
        return "No nutritional information available"
    
    formatted_text = "Nutritional Information:\n\n"
    formatted_text += f"Calories      : {nutrition_info['calories']:.2f} kcal\n"
    formatted_text += f"Fat           : {nutrition_info['fat']:.2f} g\n"
    formatted_text += f"Carbohydrates : {nutrition_info['carbohydrates']:.2f} g\n"
    formatted_text += f"Protein       : {nutrition_info['protein']:.2f} g\n"
    formatted_text += f"Calcium       : {nutrition_info['calcium']:.4f} g\n"
    formatted_text += f"Iron          : {nutrition_info['iron']:.6f} g\n"
    formatted_text += f"Magnesium     : {nutrition_info['magnesium']:.4f} g\n"
    formatted_text += f"Potassium     : {nutrition_info['potassium']:.4f} g\n"
    formatted_text += f"Sodium        : {nutrition_info['sodium']:.4f} g\n"
    formatted_text += f"Vitamin D     : {nutrition_info['vitamin_d']:.8f} g\n"
    formatted_text += f"Vitamin B12   : {nutrition_info['vitamin_b12']:.8f} g"
    
    return formatted_text


# In[8]:


def calculate_nutrition(detection_results):
    total_nutrition = {
        'calories': 0,
        'fat': 0,
        'carbohydrates': 0,
        'protein': 0,
        'calcium': 0,
        'iron': 0,
        'magnesium': 0,
        'potassium': 0,
        'sodium': 0,
        'vitamin_d': 0,
        'vitamin_b12': 0
    }

    for result in detection_results:
        food_type = result['category'].lower().replace(' ', '-')
        matching_foods = nutrition_data[nutrition_data['food_type'] == food_type]

        if not matching_foods.empty:
            food_data = matching_foods.iloc[0]
            for nutrient in total_nutrition.keys():
                if nutrient in food_data:
                    total_nutrition[nutrient] += food_data[nutrient]

    return total_nutrition


# # Format Detection Results

# In[9]:


def format_detection_results(results):
    if not results:
        return "No food detected"

    text_output = "Detected Foods:\n"
    for i, result in enumerate(results, 1):
        text_output += f"{i}. {result['category']} (Confidence: {result['confidence']:.2f})\n"

    return text_output


# # Sliding Window Food Detection

# In[10]:


def detect_food_in_image(image):
    image = np.array(image.convert("RGB"))
    original_image = image.copy()
    results = []

    window_sizes = [
        (image.shape[1] // 2, image.shape[0] // 2),
        (image.shape[1] // 3, image.shape[0] // 3),
        (image.shape[1] // 4, image.shape[0] // 4)
    ]
    step_divisors = [2, 4, 8, 12]

    all_windows = []
    all_coords = []

    for win_w, win_h in window_sizes:
        if win_w < 50 or win_h < 50:
            continue
        for divisor in step_divisors:
            step_size = max(1, min(image.shape[0], image.shape[1]) // divisor)
            for y in range(0, image.shape[0] - win_h + 1, step_size):
                for x in range(0, image.shape[1] - win_w + 1, step_size):
                    window = original_image[y:y+win_h, x:x+win_w]
                    resized = cv2.resize(window, (IMG_SIZE, IMG_SIZE))
                    input_img = preprocess_input(resized.astype(np.float32))
                    all_windows.append(input_img)
                    all_coords.append((x, y, win_w, win_h))

    if not all_windows:
        return Image.fromarray(original_image), "No food detected", {}

    all_windows = np.stack(all_windows)
    batch_size = 64
    predictions = []
    for i in range(0, len(all_windows), batch_size):
        batch = all_windows[i:i+batch_size]
        preds = model.predict(batch, verbose=0)
        predictions.extend(preds)

    sliding_window_results = []
    for pred, (x, y, w, h) in zip(predictions, all_coords):
        class_idx = np.argmax(pred)
        confidence = pred[class_idx]
        if confidence < 0.9:
            continue
        predicted_class = label_encoder.inverse_transform([class_idx])[0]
        category_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == predicted_class), "Unknown")
        sliding_window_results.append({
            'class_idx': class_idx,
            'category': category_name,
            'confidence': float(confidence),
            'bbox': [x, y, w, h]
        })

    sliding_window_results.sort(key=lambda x: x['confidence'], reverse=True)
    keep_results = []

    for current in sliding_window_results:
        x1, y1, w1, h1 = current['bbox']
        x2, y2 = x1 + w1, y1 + h1
        overlaps = False
        for kept in keep_results:
            x3, y3, w2, h2 = kept['bbox']
            x4, y4 = x3 + w2, y3 + h2
            inter_x = max(0, min(x2, x4) - max(x1, x3))
            inter_y = max(0, min(y2, y4) - max(y1, y3))
            intersection = inter_x * inter_y
            union = w1 * h1 + w2 * h2 - intersection
            iou = intersection / union if union > 0 else 0
            if iou > 0.1 and current['category'] == kept['category']:
                overlaps = True
                break
        if not overlaps:
            keep_results.append(current)

    for result in keep_results:
        x, y, w, h = result['bbox']
        category_name = result['category']
        confidence = result['confidence']
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(original_image, f"{category_name} ({confidence:.2f})",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        results.append({
            'category': result['category'],
            'confidence': result['confidence']
        })

    detection_text = format_detection_results(results)
    nutrition_info = calculate_nutrition(results)
    nutrition_text = format_nutrition_info(nutrition_info)

    return Image.fromarray(original_image), detection_text, nutrition_text


# # Gradio UI

# In[11]:


import gradio as gr

custom_theme = gr.themes.Base(
    font=gr.themes.GoogleFont("Poppins"),
    primary_hue="teal"
)

interface = gr.Interface(
    fn=detect_food_in_image,
    inputs=gr.Image(type="pil", label="📤 Upload Your Food Image"),
    outputs=[
        gr.Image(label="📷 Original Image"),
        gr.Textbox(label="🍽️ Detected Foods"),
        gr.Textbox(label="📊 Nutritional Information", elem_classes="monospace-output") 
    ],
    title="🍱 Food Detection with Nutritional Analysis",
    description=(
        "Upload a food image to detect items using a custom-trained EfficientNetB0 model. "
        "The system will identify food types and estimate total nutritional values (calories, protein, etc)."
    ),
    theme=custom_theme,
    flagging_mode="never",
    css=".monospace-output textarea { font-family: monospace !important; white-space: pre !important; }"
)


# # Launch App

# In[12]:


interface.launch(share=True)

