import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import os

# ============================
# CONFIGURATION
# ============================
BASE_SPLIT_DIR = Path(r"D:/FireData/split_dataset")
OUTPUT_DIR = Path(r"D:/FireData/output_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 20

# ============================
# HELPER FUNCTION TO LOAD MULTIPLE DATASETS
# ============================
def load_dataset(paths, split_name):
    datasets = []
    class_names = None
    for path in paths:
        ds_path = Path(path) / split_name
        if not ds_path.exists():
            continue
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            ds_path,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="int"
        )
        datasets.append(ds)
        if class_names is None:
            class_names = ds.class_names
    if len(datasets) == 0:
        raise ValueError(f"No datasets found for split '{split_name}'")
    merged_ds = datasets[0].concatenate(datasets[1]) if len(datasets) > 1 else datasets[0]
    return merged_ds, class_names

# ============================
# DEFINE PATHS
# ============================
dataset_paths = [
    BASE_SPLIT_DIR / "fireDataset_split",
    BASE_SPLIT_DIR / "fireAndSmoke_split"
]

# ============================
# LOAD TRAIN / VAL / TEST
# ============================
# Load raw datasets first (keep class_names)
raw_train_ds, class_names = load_dataset(dataset_paths, "train")
raw_val_ds, _   = load_dataset(dataset_paths, "val")
raw_test_ds, _  = load_dataset(dataset_paths, "test")

# Capture class names from the first dataset (they should be identical for merged datasets)
NUM_CLASSES = len(class_names)
print("Detected classes:", class_names)

# Apply prefetch/shuffle
AUTOTUNE = tf.data.AUTOTUNE
train_ds = raw_train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = raw_val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds  = raw_test_ds.prefetch(buffer_size=AUTOTUNE)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=IMG_SIZE + (3,)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================
# TRAIN MODEL
# ============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ============================
# EVALUATE MODEL
# ============================
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ Test accuracy: {test_acc:.3f}")

# ============================
# DETAILED CLASSIFICATION REPORT
# ============================
# Get true and predicted labels
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = np.argmax(model.predict(test_ds), axis=1)

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
print("\n📊 Classification Report:")
print(report)

# Save report to file
with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black",
             fontsize=8)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.show()

# ============================
# SAVE MODEL
# ============================
model.save(OUTPUT_DIR / "fire_detector_full.h5")

# ============================
# CONVERT TO TFLITE (quantized)
# ============================
def representative_data_gen():
    for images, _ in train_ds.take(100):
        yield [images]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_types = [tf.int8]
tflite_quant_model = converter.convert()

with open(OUTPUT_DIR / "fire_detector_int8.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("\n🎯 All models and evaluation files saved to:", OUTPUT_DIR)
