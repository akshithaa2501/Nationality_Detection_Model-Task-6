import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -----------------------
# Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "dataset", "FairFace", "fairface_label_train.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "FairFace")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "age_model.keras")

IMG_SIZE = (224, 224)

# -----------------------
# Read CSV
# -----------------------
df = pd.read_csv(CSV_PATH)

print(df.head())

# -----------------------
# Age Classes
# -----------------------
age_classes = sorted(df["age"].unique())

print("\nAge Classes:")
print(age_classes)

age_to_index = {age: idx for idx, age in enumerate(age_classes)}

# -----------------------
# Load Images
# -----------------------
X = []
Y = []

print("\nLoading images...")

for i, row in df.iterrows():

    img_path = os.path.join(IMAGE_DIR, row["file"])

    if os.path.exists(img_path):

        img = load_img(img_path, target_size=IMG_SIZE)
        img = img_to_array(img) / 255.0

        X.append(img)
        Y.append(age_to_index[row["age"]])

    if (i + 1) % 5000 == 0:
        print(f"{i+1} images processed")

X = np.array(X, dtype=np.float32)
Y = tf.keras.utils.to_categorical(Y, len(age_classes))

print("\nDataset Shape:", X.shape)

# -----------------------
# Build Model
# -----------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(len(age_classes), activation="softmax")(x)

model = Model(base_model.input, output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------
# Train
# -----------------------
model.fit(
    X,
    Y,
    batch_size=64,
    epochs=3,
    validation_split=0.2
)

# -----------------------
# Save
# -----------------------
model.save(MODEL_SAVE_PATH)

print("\nAge model saved successfully!")
print(MODEL_SAVE_PATH)