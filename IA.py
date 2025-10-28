import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf

# 1) Caricamento dati 

# Percorsi cartelle
folder_path_high = "./Dataset/high"
folder_path_medium = "./Dataset/medium"
folder_path_low = "./Dataset/low"
folder_path_unlabeled = "./Dataset/unlabeled"

# Trova file CSV
csv_files_high = [f for f in os.listdir(folder_path_high) if f.endswith(".csv")]
csv_files_medium = [f for f in os.listdir(folder_path_medium) if f.endswith(".csv")]
csv_files_low = [f for f in os.listdir(folder_path_low) if f.endswith(".csv")]
csv_files_unlabeled = [f for f in os.listdir(folder_path_unlabeled) if f.endswith(".csv")]

# Concatena
df_high = pd.concat([pd.read_csv(os.path.join(folder_path_high, f)) for f in csv_files_high], ignore_index=True)
df_medium = pd.concat([pd.read_csv(os.path.join(folder_path_medium, f)) for f in csv_files_medium], ignore_index=True)
df_low = pd.concat([pd.read_csv(os.path.join(folder_path_low, f)) for f in csv_files_low], ignore_index=True)

# Etichette
df_high["label"] = "high"
df_medium["label"] = "medium"
df_low["label"] = "low"

df = pd.concat([df_high, df_medium, df_low], ignore_index=True)  # escludi unlabeled per l'addestramento

#  2) Encoding 
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

#  3) Feature scaling 
X = df.drop(columns=["label"])
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  4) Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#  5) Modello Keras 
num_features = X_train.shape[1]
num_classes = len(np.unique(y_train))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_features,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#  6) Addestramento 
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1)

#  7) Valutazione
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuratezza del modello: {accuracy * 100:.2f}%")

#  8) Salvataggio del modello
model.save("model.h5")

# 9) Salvataggio LabelEncoder e scaler
import joblib
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")