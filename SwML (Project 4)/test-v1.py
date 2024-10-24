import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import os

model = load_model('MODEL1.keras')

data = pd.read_csv('test_df.csv')  
X_test = data['filepaths'].values
y_test = data['labels'].values

print("Original labels in test data:")
print(y_test[:10])

print("Unique labels in test data:", np.unique(y_test))

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    img = load_img(image_path, target_size=(128, 128))  
    img_array = img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

X_test_processed = []
valid_image_paths = []
for img_path in X_test:
    print(f"Processing image: {img_path}")  
    try:
        processed_image = preprocess_image(img_path)
        X_test_processed.append(processed_image)
        valid_image_paths.append(img_path)
    except FileNotFoundError as e:
        print(e)
        continue

if not X_test_processed:
    raise ValueError("No valid images found. Ensure the file paths are correct.")

X_test_processed = np.vstack(X_test_processed)

y_test_filtered = [label for img_path, label in zip(X_test, y_test) if img_path in valid_image_paths]

y_pred = model.predict(X_test_processed)
y_pred_classes = np.argmax(y_pred, axis=1)

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
y_pred_labels = [class_names[i] for i in y_pred_classes]

print("Sample predictions:")
for i in range(10):
    print(f"True label: {y_test_filtered[i]}, Predicted: {y_pred_labels[i]}")


report = classification_report(y_test_filtered, y_pred_labels, target_names=class_names)
print(report)


cm = confusion_matrix(y_test_filtered, y_pred_labels, labels=class_names)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)


plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


print("Checking a smaller subset of data for manual verification...")
sample_size = 10
sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
X_sample = [X_test[i] for i in sample_indices]
y_sample = [y_test[i] for i in sample_indices]

X_sample_processed = []
for img_path in X_sample:
    try:
        processed_image = preprocess_image(img_path)
        X_sample_processed.append(processed_image)
    except FileNotFoundError as e:
        print(e)
        continue

if X_sample_processed:
    X_sample_processed = np.vstack(X_sample_processed)
    y_sample_pred = model.predict(X_sample_processed)
    y_sample_pred_classes = np.argmax(y_sample_pred, axis=1)
    y_sample_pred_labels = [class_names[i] for i in y_sample_pred_classes]

    print("Sample predictions for manual verification:")
    for i in range(len(X_sample)):
        print(f"File: {X_sample[i]}, True label: {y_sample[i]}, Predicted: {y_sample_pred_labels[i]}")
else:
    print("No valid images found in the smaller subset.")
