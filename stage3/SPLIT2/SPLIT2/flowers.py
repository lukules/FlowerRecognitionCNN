import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

data_dir = 'flowers'
img_size = (128, 128)  # Ensure numpy arrays are reshaped to this size if not already
batch_size = 16
channels = 3
img_shape = (img_size[0], img_size[1], channels)

def generate_data_paths(data_dir):
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            if file.endswith('.npy'):
                fpath = os.path.join(foldpath, file)
                filepaths.append(fpath)
                labels.append(fold)
    return filepaths, labels

filepaths, labels = generate_data_paths(data_dir)
df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})

# Create consistent one-hot encoding for labels
label_encoder = pd.get_dummies(df['labels']).columns
def encode_labels(labels):
    return pd.get_dummies(labels, columns=label_encoder).reindex(columns=label_encoder, fill_value=0).values

train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123)
valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123)

def load_npy(file_path, label):
    image = np.load(file_path)  # Load the numpy array from .npy file
    image = np.resize(image, img_shape)  # Resize to expected shape
    return image, label

def data_generator(df, batch_size=32):
    while True:
        # shuffle dataframe
        df = df.sample(frac=1).reset_index(drop=True)
        for i in range(0, df.shape[0], batch_size):
            batch_df = df.iloc[i:i + batch_size]
            images, labels = zip(*[load_npy(fp, lb) for fp, lb in zip(batch_df['filepaths'], batch_df['labels'])])
            images = np.array(images)
            labels = encode_labels(labels)  # Use consistent encoding
            yield images, labels

# Prepare data generators
train_gen = data_generator(train_df, batch_size)
valid_gen = data_generator(valid_df, batch_size)

# Building the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=img_shape),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(len(label_encoder), activation='softmax')  # Dynamically adjust output layer based on unique labels
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

csv_logger = CSVLogger('flowers-training.log', separator=',', append=False)
epochs = 20

checkpoint_filepath = 'best_model.keras'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

# Fit model
history = model.fit(x=train_gen,
                    steps_per_epoch=len(train_df)//batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=valid_gen,
                    validation_steps=len(valid_df)//batch_size,
                    callbacks=[csv_logger, model_checkpoint_callback])

model.save('flowers-model.keras')

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.show()
