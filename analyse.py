import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import joblib
from tensorflow.keras.preprocessing import image


fruit_model = tf.keras.models.load_model("leaf_plant_model.h5")
disease_model = tf.keras.models.load_model("fruit_disease_predictor.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")


class_names = ["Apple", "Banana", "Grape", "Watermelon"]
disease_classes = {
    "Apple": {
        "apple_scab": "Apply fungicides containing copper or sulfur. Prune infected leaves.",
        "applerustleaves": "Use rust-resistant apple varieties. Remove and destroy infected leaves."
    },
    "Banana": {
        "banana_anthracnose": "Apply a copper-based fungicide. Keep leaves dry to prevent spread.",
        "banana_sigatoka_leaf_spot": "Use fungicides like Mancozeb. Improve airflow around plants."
    },
    "Grape": {
        "grape_downey_mildew": "Apply a Bordeaux mixture (copper sulfate). Ensure good drainage.",
        "grape_phomopsis": "Use a fungicide spray in early spring. Remove infected canes."
    },
    "Watermelon": {
        "Watermelon_gummy_stem_blight": "Apply chlorothalonil fungicide. Avoid excessive watering.",
        "Watermelon_leaf_spots": "Use a sulfur-based fungicide. Rotate crops annually."
    }
}


ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

root = ctk.CTk()
root.title("Plant Disease Predictor")
root.geometry("750x600")
root.configure(bg="#E8F5E9")


main_frame = ctk.CTkFrame(root, fg_color="#E8F5E9")
main_frame.pack(expand=True)


logo_path = r"C:\Users\USER\Downloads\leaf_logo.png"
try:
    logo = ctk.CTkImage(Image.open(logo_path), size=(80, 80))
    logo_label = ctk.CTkLabel(main_frame, image=logo, text="")
    logo_label.pack(pady=10)
except Exception as e:
    print(f"Error loading logo: {e}")


def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((180, 180))
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img
        predict_fruit(file_path)


def predict_fruit(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = fruit_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    fruit_name = class_names[predicted_class_index]
    result_label.configure(text=f"Detected Fruit: {fruit_name}")
    predict_disease(fruit_name)


def predict_disease(fruit):
    temperature = float(temp_entry.get())
    humidity = float(humidity_entry.get())
    soil_moisture = float(soil_entry.get())

    input_data = scaler.transform([[temperature, humidity, soil_moisture]])
    predictions = disease_model.predict(input_data)[0]

    fruit_diseases = disease_classes.get(fruit, {})
    disease_indices = [list(label_encoder.classes_).index(d) for d in fruit_diseases.keys()]
    filtered_predictions = predictions[disease_indices]

    max_index = np.argmax(filtered_predictions)
    predicted_disease = list(fruit_diseases.keys())[max_index]
    prevention_tip = fruit_diseases[predicted_disease]

    disease_label.configure(text=f"Predicted Disease: {predicted_disease}")
    prevention_label.configure(text=f"Prevention: {prevention_tip}")


upload_button = ctk.CTkButton(main_frame, text="Upload Image", command=upload_image, fg_color="#4CAF50",
                              hover_color="#388E3C", corner_radius=25, width=250)
upload_button.pack(pady=10)


image_label = ctk.CTkLabel(main_frame, text="No Image Uploaded", width=180, height=180, fg_color="#C8E6C9")
image_label.pack(pady=10)


input_frame = ctk.CTkFrame(main_frame, fg_color="#E8F5E9")
input_frame.pack(pady=10)

temp_entry = ctk.CTkEntry(input_frame, placeholder_text="Temperature (°C)", width=150)
temp_entry.grid(row=0, column=0, padx=5)

humidity_entry = ctk.CTkEntry(input_frame, placeholder_text="Humidity (%)", width=150)
humidity_entry.grid(row=0, column=1, padx=5)

soil_entry = ctk.CTkEntry(input_frame, placeholder_text="Soil Moisture (%)", width=150)
soil_entry.grid(row=0, column=2, padx=5)


predict_button = ctk.CTkButton(main_frame, text="Predict Disease",
                               command=lambda: predict_disease(result_label.cget("text").split(": ")[1]),
                               fg_color="#FF9800", hover_color="#F57C00", corner_radius=25, width=250)
predict_button.pack(pady=10)


result_label = ctk.CTkLabel(main_frame, text="", font=("Arial", 16, "bold"), justify="center")
result_label.pack(pady=10)

disease_label = ctk.CTkLabel(main_frame, text="", font=("Arial", 16, "bold"), justify="center")
disease_label.pack(pady=10)

prevention_label = ctk.CTkLabel(main_frame, text="", font=("Arial", 14), wraplength=500, justify="center")
prevention_label.pack(pady=10)

root.mainloop()
