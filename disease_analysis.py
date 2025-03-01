import tensorflow as tf
import numpy as np
import sqlite3
import os
from tensorflow.keras.preprocessing import image
import joblib  # To load scaler

# Load models
model1_path = "leaf_plant_model.h5"  # Fruit classifier model
model2_path = "fruit_disease_predictor.h5"  # Disease prediction model
scaler_path = "scaler.pkl"  # Scaler used during training

model1 = tf.keras.models.load_model(model1_path)
model2 = tf.keras.models.load_model(model2_path)
scaler = joblib.load(scaler_path)  # Load the scaler


fruit_classes = ["Apple", "Banana", "Grape", "Watermelon"]


disease_classes = {
    "Apple": ["apple_scab", "applerustleaves"],
    "Banana": ["banana_anthracnose", "banana_sigatoka_leaf_spot"],
    "Grape": ["grape_downey_mildew", "grape_phomopsis"],
    "Watermelon": ["Watermelon_gummy_stem_blight", "Watermelon_leaf_spots"]
}


def preprocess_image(img_path):
    """Load and preprocess the image"""
    img = image.load_img(img_path, target_size=(150, 150))  # Resize to match training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array


def predict_fruit(img_path):
    """Predicts if the image is Apple, Banana, Grape, or Watermelon"""
    img_array = preprocess_image(img_path)
    predictions = model1.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    if predicted_class_index >= len(fruit_classes):
        print("Error: Invalid fruit class index!")
        return None

    predicted_fruit = fruit_classes[predicted_class_index]
    confidence = np.max(predictions) * 100
    print(f"Predicted Fruit: {predicted_fruit} (Confidence: {confidence:.2f}%)")

    return predicted_fruit


def get_sensor_data(fruit):
    """Fetch Temperature, Humidity, Soil Moisture for the given fruit"""
    db_path = fruit.lower() + ".db"  # Example: apple.db, banana.db, etc.

    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found!")
        return None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    disease_tables = disease_classes.get(fruit, [])

    sensor_data = None
    for table in disease_tables:
        query = f"SELECT Temperature, Humidity, Soil_Moisture FROM {table} LIMIT 1"
        try:
            cursor.execute(query)
            row = cursor.fetchone()
            if row:
                sensor_data = np.array(row).reshape(1, -1)
                break  # Use the first valid entry found
        except Exception as e:
            print(f"Error fetching data from {table}: {e}")

    conn.close()

    if sensor_data is not None:
        return sensor_data
    else:
        print("No valid sensor data found!")
        return None


def predict_disease(fruit):
    """Predicts disease based on Temperature, Humidity & Soil Moisture"""
    sensor_data = get_sensor_data(fruit)
    if sensor_data is None:
        print("Skipping disease prediction due to missing sensor data.")
        return


    sensor_data_scaled = scaler.transform(sensor_data)


    disease_prediction = model2.predict(sensor_data_scaled)
    predicted_disease_index = np.argmax(disease_prediction, axis=1)[0]

    disease_list = disease_classes.get(fruit, [])

    if predicted_disease_index >= len(disease_list):
        print("Error: Predicted disease index is out of range!")
        return

    predicted_disease = disease_list[predicted_disease_index]
    print(f"Predicted Disease: {predicted_disease}")


# Pipeline Execution
if __name__ == "__main__":
    img_path = input("Enter the path of the image: ")
    if os.path.exists(img_path):
        fruit = predict_fruit(img_path)
        if fruit:
            predict_disease(fruit)
    else:
        print("Invalid image path!")
