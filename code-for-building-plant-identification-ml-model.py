import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# ✅ Define dataset paths
train_data_dir = 'data/train'  # Folder with training images (organized by class)
validation_data_dir = 'data/validation'  # Folder with validation images

# ✅ Image size and batch size
img_width, img_height = 150, 150
batch_size = 32

# ✅ Data Augmentation & Normalization
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

# ✅ Load training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# ✅ Load validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# ✅ Load Pretrained MobileNetV2 Model (Better Accuracy)
base_model = MobileNetV2(input_shape=(img_width, img_height, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze base model

# ✅ Define New Model for Plant Identification
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# ✅ Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train the Model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=50
)

# ✅ Save the Model for Future Use
model.save("leaf_plant_model.h5")

# ✅ Function to Make Predictions on New Leaf Images
def predict_leaf(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = list(train_generator.class_indices.keys())[predicted_class_index]

    return predicted_class_name

test_image_path = "test_leaf.jpg"  # Replace with your test image
predicted_class = predict_leaf(test_image_path)
print(f"Predicted Plant Type: {predicted_class}")
