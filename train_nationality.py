import os
import cv2
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------------------------------
# Project Paths
# ----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAIRFACE_DIR = os.path.join(BASE_DIR, "dataset", "FairFace")

TRAIN_CSV = os.path.join(FAIRFACE_DIR, "fairface_label_train.csv")
VAL_CSV = os.path.join(FAIRFACE_DIR, "fairface_label_val.csv")

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_CLASSES = 4
# ----------------------------------------------------
# Read Dataset
# ----------------------------------------------------

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

# Convert FairFace race into 4 nationality classes
def race_to_nationality(race):

    if race == "Indian":
        return "Indian"

    elif race == "White":
        return "United States"

    elif race == "Black":
        return "African"

    else:
        return "Others"


train_df["nationality"] = train_df["race"].apply(race_to_nationality)
val_df["nationality"] = val_df["race"].apply(race_to_nationality)


CLASS_NAMES = [
    "Indian",
    "United States",
    "African",
    "Others"
]

label_map = {
    "Indian": 0,
    "United States": 1,
    "African": 2,
    "Others": 3
}

train_df["label"] = train_df["nationality"].map(label_map)
val_df["label"] = val_df["nationality"].map(label_map)

print("Training images :", len(train_df))
print("Validation images :", len(val_df))

print("\nClass Distribution\n")
print(train_df["nationality"].value_counts())

print("\nSample Data")
print(train_df.head())

# ----------------------------------------------------
# Verify Image Paths
# ----------------------------------------------------

train_df["image_path"] = train_df["file"].apply(
    lambda x: os.path.join(FAIRFACE_DIR, x)
)

val_df["image_path"] = val_df["file"].apply(
    lambda x: os.path.join(FAIRFACE_DIR, x)
)


# Keep only existing images
train_df = train_df[train_df["image_path"].apply(os.path.exists)].reset_index(drop=True)
val_df = val_df[val_df["image_path"].apply(os.path.exists)].reset_index(drop=True)


print("\n==========================================")
print("Training Images :", len(train_df))
print("Validation Images :", len(val_df))
print("==========================================")

if len(train_df) == 0:
    raise Exception("Training images not found!")

if len(val_df) == 0:
    raise Exception("Validation images not found!")

print("\nDataset Found Successfully!")

print("\nSample Image:")
print(train_df.iloc[0]["image_path"])

# ----------------------------------------------------
# Custom Data Generator
# ----------------------------------------------------

class DataGenerator(Sequence):

    def __init__(self, dataframe, batch_size=BATCH_SIZE, shuffle=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.dataframe) / self.batch_size)

    def __getitem__(self, index):

        batch_indexes = self.indexes[
            index * self.batch_size:(index + 1) * self.batch_size
        ]

        batch_df = self.dataframe.iloc[batch_indexes]

        images = []
        labels = []

        for _, row in batch_df.iterrows():

            image = cv2.imread(row["image_path"])

            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = image.astype("float32") / 255.0

            images.append(image)
            labels.append(row["label"])

        images = np.array(images, dtype=np.float32)
        labels = tf.keras.utils.to_categorical(labels, NUM_CLASSES)

        return images, labels

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indexes)
# ----------------------------------------------------
# Create Data Generators
# ----------------------------------------------------

train_generator = DataGenerator(
    train_df,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_generator = DataGenerator(
    val_df,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nData generators created successfully.")


# ----------------------------------------------------
# Build MobileNetV2 Model
# ----------------------------------------------------

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary\n")
model.summary()

# ----------------------------------------------------
# Compute Class Weights
# ----------------------------------------------------

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)

class_weights = dict(enumerate(class_weights))

print("\nClass Weights")
print(class_weights)


# ----------------------------------------------------
# Callbacks
# ----------------------------------------------------

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "nationality_model.keras")

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


# ----------------------------------------------------
# Train Model
# ----------------------------------------------------

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop],
    verbose=1
)


# ----------------------------------------------------
# Evaluate Model
# ----------------------------------------------------

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(val_generator, verbose=1)

print(f"\nValidation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy*100:.2f}%")


# ----------------------------------------------------
# Save Final Model
# ----------------------------------------------------

FINAL_MODEL = os.path.join(
    MODEL_DIR,
    "nationality_model_final.keras"
)

model.save(FINAL_MODEL)

print("\nFinal model saved successfully.")
print("Saved at :", FINAL_MODEL)

print("\nTraining Completed Successfully!")

# ----------------------------------------------------
# Main
# ----------------------------------------------------

if __name__ == "__main__":
    pass