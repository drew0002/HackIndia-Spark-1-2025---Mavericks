
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib


fruit_dbs = ["apple.db", "banana.db", "grapes.db", "watermelon.db"]
all_data = []

for db in fruit_dbs:
    conn = sqlite3.connect(db)
    cursor = conn.cursor()


    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_sequence';")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]


        query = f"SELECT Temperature, Humidity, Soil_Moisture, '{table_name}' AS disease FROM {table_name}"

        try:
            df = pd.read_sql_query(query, conn)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading table {table_name}: {e}")

    conn.close()


data = pd.concat(all_data, ignore_index=True)


label_encoder = LabelEncoder()
data["disease"] = label_encoder.fit_transform(data["disease"])
joblib.dump(label_encoder, "label_encoder.pkl")  # Save the label encoder


X = data[["Temperature", "Humidity", "Soil_Moisture"]]
y = data["disease"]


scaler = MinMaxScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')  #
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))


model.save("fruit_disease_predictor.h5")


print("Label Mapping:", dict(enumerate(label_encoder.classes_)))

