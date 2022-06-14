import os
import shutil

import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.cluster import KMeans
from tqdm import tqdm
import segmentation_models as sm


def removeIncompleteRuns(path):
    for root, runs, files in os.walk(path):
        for run in runs:
            run = os.path.join(root, run)
            if len(next(os.walk(os.path.join(run, 'model_data')))[2]) == 0:
                shutil.rmtree(run, ignore_errors=True)
        break


def initializeDirectories(run_dir, tmp_data_path, results_path, model_data_path, log_path, last_model_data_path):
    removeIncompleteRuns(run_dir)

    os.makedirs(tmp_data_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(model_data_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

# It loads the images by using OpenCV library
def images_upload(_path):
    imgs = []
    for root, subfolders, files in os.walk(_path):
        for file in tqdm(files):
            filename = root + os.sep + file
            if filename.endswith('jpg') or filename.endwith('png'):
                imgs.append(cv2.imread(filename, cv2.COLOR_BGR2RGB))
    return imgs


# Splitting the original dataset image into two images: the input and the desired label
def split_input_mask(_images, IMG_HEIGHT):
    _inputs = []
    _masks = []
    for img in _images:
        sx = img[:, :IMG_HEIGHT]
        _inputs.append(sx)
        dx = img[:, IMG_HEIGHT:]
        _masks.append(dx)
    return _inputs, _masks


# New labels are created by using kMeans with k=10 and this model is fitted into random color array then
# a label prediction is made to the index image in masks. This prediction array will be our desired output labels
# for each image, thus the ground of truth to be used in the U-Net training phase.
# Clearly kMeans could mislead clustering since its initialization is random, so labels can change in each run.
def new_labels(_masks, IMG_DIMENSIONS, num_classes=10):
    color_array = np.random.choice(range(256), IMG_DIMENSIONS[0] * IMG_DIMENSIONS[1] * IMG_DIMENSIONS[2]).reshape(-1, 3)
    label_model = KMeans(n_clusters=num_classes)
    label_model.fit(color_array)
    _labels = []
    for index in tqdm(range(len(_masks))):
        _labels.append(label_model.predict(_masks[index].reshape(-1, 3)).reshape(256, 256))
    return _labels


def convert_img2float(data):
    converted = []
    for d in tqdm(data):
        img = tf.image.convert_image_dtype(d, tf.float32)
        del d
        converted.append(img)
    return converted

# Functions to define whaat model using in this project, there are several options that can be set.
# In the comments are reported some tips taken from Tensorflow and Segmentation Models guidelines.
def build_model(num_classes=10):
    # Input size: 256x256x3

    # Default U-Net model
    # _model = sm.Unet()

    # If you want to change the network architecture by choosing backbones with fewer or more parameters and
    # use pretrained weights to initialize it
    # _model = sm.Unet('resnet34', encoder_weights='imagenet')

    _model = sm.Unet('resnet34', classes=num_classes, activation='softmax')
    return _model

# Callbacks to be used during the training model. Early Stopping is used to avoid overfitting situations.
# Model Checkpoint is useful to save the best weight configuration of the NNetwork thus we are able to load it again
# in the next runs.
# TensorBoard actually is useless but originally used to try to save all training stats to recover some intel.
def trainingCallbacks(checkpoint_path, log_path, patience=5):
    ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1,
                                                         save_best_only=True,
                                                         save_weights_only=True)
    EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)
    TensorBoard = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0,
                                                 write_graph=True,
                                                 write_images=True,
                                                 write_steps_per_second=False,
                                                 update_freq="epoch")
    return [ModelCheckpoint, EarlyStopping, TensorBoard]

# Dummy function that plot some images: the input, the original mask and the labelled mask
def show_some_labels(inputs, masks, labels, results_path):
    plt.figure(figsize=(15, 15))
    fig, axes = plt.subplots(3, 3)
    for r in range(3):
        index = np.random.randint(0, len(labels))
        axes[r][0].imshow(inputs[index])
        axes[r][0].set_title("Original")
        axes[r][0].axis('off')
        axes[r][1].imshow(masks[index])
        axes[r][1].set_title("Mask")
        axes[r][1].axis('off')
        axes[r][2].imshow(labels[index])
        axes[r][2].set_title("Labelled Mask")
        axes[r][2].axis('off')
    plt.savefig(results_path + 'example_org-mask-label.png', dpi=300)
    plt.show()


def show_images(data):
    plt.figure(figsize=(10, 10))
    for index in range(9):
        plt.subplot(3, 3, index + 1)
        plt.imshow(data[np.random.randint(0, len(data))])
        plt.axis('off')
    plt.show()

# It randomly take the index of an input and a mask and put them aside into a plot
def images_compare(_inputs, _masks):
    index = np.random.randint(0, len(_inputs))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(_inputs[index])
    ax1.axis('off')
    ax1.set_title('Original')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(_masks[index])
    ax2.axis('off')
    ax2.set_title('Mask')
    plt.show()

# Deprecated function to try predicting the semantic segmentation on a image coming from outside the dataset.
# I don't know why but it doesn't work since the input dimensions are not correct. That's odd since images are 256x256x3
def prova(model):
    example_input = [cv2.imread('..' + os.sep + 'Material' + os.sep + 'Other' + os.sep + 'originale_simulator.png',
                                cv2.COLOR_BGR2RGB)]
    example_mask = [
        cv2.imread('..' + os.sep + 'Material' + os.sep + 'Other' + os.sep + 'mask_simulator.png', cv2.COLOR_BGR2RGB)]
    label = new_labels(example_mask, example_mask[0].shape, 10)
    images_compare(example_mask, label)
    example_input = convert_img2float(example_input)
    print(example_input[0].shape)
    prediction = model.predict(example_input[0])
    masks_predicted = tf.argmax(prediction, axis=-1)
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    axes[0][0].imshow(example_input[0])
    axes[0][0].set_title("Original")
    axes[0][0].axis('off')
    axes[0][1].imshow(label[0])
    axes[0][1].set_title("Mask")
    axes[0][1].axis('off')
    axes[0][2].imshow(masks_predicted[0])
    axes[0][2].set_title("Prediction")
    axes[0][2].axis('off')
    plt.show()
