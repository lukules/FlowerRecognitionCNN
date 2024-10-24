import os
import numpy as np
import pandas as pd

from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


data_dir = 'flowers'

img_size = 128
batch_size = 16
test_batch_size = 8

images = []

category_to_number = {
    'daisy': 0,
    'dandelion': 1,
    'sunflower': 2,
    'rose': 3,
    'tulip': 4
}

number_to_category = {
    0: 'daisy',
    1: 'dandelion',
    2: 'sunflower',
    3: 'rose',
    4: 'tulip'
}


def generate_data_paths(data_dir):
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(category_to_number[fold])

    return filepaths, labels


filepaths, labels = generate_data_paths(data_dir)


def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((img_size, img_size))
    image = np.array(image) / 255
    return image


for i in filepaths:
    images.append(load_image(i))


def create_df(images, labels):
    Fseries = pd.Series(images, name='images')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    return df


df = create_df(images, labels)

# train
train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123)

# validation and test
valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123)


def scalar(img):
    return img


tr_gen = ImageDataGenerator(preprocessing_function=scalar,
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            brightness_range=[0.4, 0.8],
                            zoom_range=0.2,
                            horizontal_flip=True,
                            vertical_flip=True)

ts_gen = ImageDataGenerator(preprocessing_function=scalar,
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            brightness_range=[0.4, 0.8],
                            zoom_range=0.2,
                            horizontal_flip=True,
                            vertical_flip=True)

train_gen = tr_gen.flow_from_dataframe(train_df,
                                       x_col='filepaths',
                                       y_col='labels',
                                       target_size=img_size,
                                       class_mode='categorical',
                                       color_mode='rgb',
                                       shuffle=True,
                                       batch_size=batch_size)

valid_gen = ts_gen.flow_from_dataframe(valid_df,
                                       x_col='filepaths',
                                       y_col='labels',
                                       target_size=img_size,
                                       class_mode='categorical',
                                       color_mode='rgb',
                                       shuffle=True,
                                       batch_size=batch_size)

test_gen = ts_gen.flow_from_dataframe(test_df,
                                      x_col='filepaths',
                                      y_col='labels',
                                      target_size=img_size,
                                      class_mode='categorical',
                                      color_mode='rgb',
                                      shuffle=False,
                                      batch_size=test_batch_size)