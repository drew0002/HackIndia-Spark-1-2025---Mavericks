import tensorflow as tf
import numpy as np
import joblib
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler


fruit_model_path = "leaf_plant_model.h5"
fruit_model = tf.keras.models.load_model(fruit_model_path)


disease_model_path = "fruit_disease_predictor.h5"
disease_model = load_model(disease_model_path)


scaler_path = "scaler.pkl"
scaler = joblib.load(scaler_path)

label_encoder_path = "label_encoder.pkl"
label_encoder = joblib.load(label_encoder_path)


class_names = ["Apple", "Banana", "Grape", "Watermelon"]


disease_classes = {
    "Apple": ["apple_scab", "applerustleaves"],
    "Banana": ["banana_anthracnose", "banana_sigatoka_leaf_spot"],
    "Grape": ["grape_downey_mildew", "grape_phomopsis"],
    "Watermelon": ["Watermelon_gummy_stem_blight", "Watermelon_leaf_spots"]
}


def preprocess_image(img_path):
    """Load and preprocess image for fruit classification."""
    img = image.load_img(img_path, target_size=(150, 150))  # Resize to match training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array


def predict_fruit(img_path):
    """Predicts the fruit type from an image."""
    img_array = preprocess_image(img_path)
    predictions = fruit_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    if predicted_class_index >= len(class_names):
        print(f"Error: Predicted index {predicted_class_index} is out of range!")
        return None

    predicted_fruit = class_names[predicted_class_index]
    return predicted_fruit


def predict_disease(fruit, temperature, humidity, soil_moisture):
    """Predicts disease based on fruit type and sensor readings."""
    input_data = np.array([[temperature, humidity, soil_moisture]])
    input_data_scaled = scaler.transform(input_data)

    predictions = disease_model.predict(input_data_scaled)[0]


    all_diseases = list(label_encoder.classes_)


    fruit_diseases = disease_classes.get(fruit, [])


    if not fruit_diseases:
        print(f"Error: No diseases mapped for {fruit}.")
        return None


    disease_indices = [all_diseases.index(d) for d in fruit_diseases if d in all_diseases]


    if "apple_scab" not in all_diseases:
        print("Error: 'apple_scab' is not in the label encoder classes.")
        return None


    filtered_predictions = predictions[disease_indices]


    max_index = np.argmax(filtered_predictions)
    predicted_disease = fruit_diseases[max_index]
    confidence = filtered_predictions[max_index] * 100

    print(f"\nPredicted Disease: {predicted_disease} ")

    return predicted_disease



# 🎯 **Pipeline Execution**
if __name__ == "__main__":
    img_path = input("Enter the path of the fruit image: ").strip()

    if not os.path.exists(img_path):
        print("Invalid image path!")
        exit()

    fruit = predict_fruit(img_path)

    if fruit:
        temperature = float(input("Enter temperature: "))
        humidity = float(input("Enter humidity: "))
        soil_moisture = float(input("Enter soil moisture: "))

        predict_disease(fruit, temperature, humidity, soil_moisture)
