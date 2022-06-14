import os
import json
from datetime import datetime
from os.path import exists

import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models as sm
import tensorflow as tf
from numpy import load, savez_compressed
from sklearn.model_selection import train_test_split
from tensorflow import keras

import utils

# Tensorflow initializations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
keras.backend.set_image_data_format('channels_last')
sm.set_framework('tf.keras')
sm.framework()

# Dataset path building
dataset = 'CITYSCAPES'
run_dir = '.' + os.sep + 'runs' + os.sep

# Current runtime directory will be created based on actual system datetime
# This is mostly because we don't want to lose results of each run
now = datetime.now()
run_datetime = now.strftime("%d-%m-%Y_%H-%M-%S")
actual_run_dir = run_dir + run_datetime + os.sep

# In results_data will be stored all the images produced by the script of the current run
results_path = actual_run_dir + 'results' + os.sep
# In model_data_path will be stored the (trained) model data such as its configuration, its weights and history
model_data_path = actual_run_dir + 'model_data' + os.sep
# In log_path will be stored all the logs of this run
log_path = actual_run_dir + 'logs' + os.sep
# In tmp_data_path will be stored the labelled masks and the alias of the latest run's model info
tmp_data_path = '.' + os.sep + 'tmp' + os.sep
last_model_data_path = tmp_data_path + 'latest_model_data'
# Dataset is located the dataset_path
dataset_path = '..' + os.sep + 'Material' + os.sep + 'Datasets' + os.sep + dataset + os.sep

utils.initializeDirectories(last_model_data_path=last_model_data_path, log_path=log_path,
                            model_data_path=model_data_path, results_path=results_path, run_dir=run_dir,
                            tmp_data_path=tmp_data_path)

# Loading the entire dataset, no distinction is made between the validation and train subset since will be split
# later randomly
print("Loading images...")
images = None
if dataset == 'CITYSCAPES':
    images = utils.images_upload(dataset_path + os.sep + "val") + utils.images_upload(dataset_path + os.sep + "train")
print(" Done.\n")

# Images information
size_dataset = len(images)
IMG_DIMENSIONS = images[0].shape
IMG_HEIGHT = images[0].shape[0]
IMG_WIDTH = images[0].shape[1]
IMG_CHANNELS = images[0].shape[2]
num_items = IMG_WIDTH * IMG_HEIGHT

# Splitting the images from the ground truth (masks)
inputs, masks = utils.split_input_mask(images, IMG_HEIGHT)
del images

# show_images(inputs)
# show_images(masks)
# utils.images_compare(inputs, masks)

############################# Creating labels with K-means #############################
# Since the label of the dataset comes with pixel color-labeled encoded, it was necessary to reconstruct the labels
# by using the clustering on colors. The optimal algorithm chosen is K-Means with K set to 10 classes/clusters.
# The CityScapes dataset comes with 30 labels, but with K=10 classes everything works better.
# K, the number of cluster is a hyperparameter.
# Since this operation must be done at least once, if we repeat the evaluation of a model already trained,
# the results will be altered since each clustering is random. Thus, it was necessary to save the labels locally
# to be used in next runs (if we want to re-use the same trained model).
# Labels are compressed to save disk space.

num_classes = 10

if not exists(tmp_data_path + dataset + '_labels.npz'):
    print("Creating labels...")
    labels = utils.new_labels(masks, IMG_DIMENSIONS)
    print("Saving compressed labels...")
    savez_compressed(tmp_data_path + dataset + '_labels.npz', labels)
    print("Done.\n\n")
else:
    print("Loading labels...")
    labels = load(tmp_data_path + dataset + '_labels.npz')
    labels = labels['arr_0']  # Compressed numpy file is saved as array of arrays for multiple d. structures
    print("Done.\n")

# Creating a 3x3 grid of original image, its original mask and the new mask computed by K-Means.
utils.show_some_labels(inputs, masks, labels, results_path)

# Converting the images into ndArrays, compatible with TensorFlow data manipulation specifications.
print("Converting images to float type...")
inputs = utils.convert_img2float(inputs)
print("Done.\n\n")

############################################# U-Net Model Building ############################################
# We are gonna create a new U-Net model based on Resnet34 backbone, suggested into Segmentation Models, which is the
# package that includes the most known models for Segmentation problems, like semantic segmentation in this case.
# Segmentation models is used to save time and model creation.
# Unfortunately the plot model is not simple as it should be. My guess is that maybe because it depends on
# the backbone configuration.
# Softmax layer is used as activation function.
print("Building Model...")
model = utils.build_model(num_classes)
tf.keras.utils.plot_model(model, to_file=os.path.join(model_data_path, 'model_architecture.png'), show_shapes=True,
                          show_layer_names=True)
print("Done.\n")

print("Compiling model...")
# As suggested by Tensorflow tutorials, the sparse categorical cross-entropy loss function can be used only when
# we are using a SoftMax activation function.
# Adam seems to be the best optimizer function. The learning rate is the default one suggested in
# Tensorflow tutorial page.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
print("Done.\n")

############################################## Model Training ######################################################
# Splits 'inputs' and 'labels' arrays randomly by returning a new set for testing which is the 20% of total.
test_split = 0.2
val_split = 0.2
print("Processing Dataset...")
inputs_train, inputs_test, labels_train, labels_test = train_test_split(np.array(inputs), np.array(labels),
                                                                        test_size=test_split)
del inputs, labels
print("Done.\n")
# Computing and displaying some dataset stats
size_val_set = len(inputs_train) * val_split
size_test_set = len(inputs_test) * test_split
size_train_set = len(inputs_train) - size_val_set

print("######## DATASET INFO ########")
print(f"Train set size: {size_train_set}")
print(f"Validation set size: {size_val_set}")
print(f"Test set size: {size_test_set}")
print("------------------------------")
print(f"Total size: {size_dataset}\n")

print("Evaluating untrained model...")
# Evaluate the model with Test set BEFORE the training. This is interesting because randomness sometimes
# gave in some run an accuracy of 30% on untrained model. Probably most of this it comes from the pre-trained weights
# of the chosen Resnet backbone.
pre_loss, pre_acc = model.evaluate(inputs_test, labels_test, verbose=2)
print("Untrained model, accuracy: {:5.2f}% \n".format(100 * pre_acc))


# Few tricks and checks to use pre trained data on successive runs
hist = None
weights_exists = False
model_exists = False
hist_exists = False

if os.path.islink(last_model_data_path):
    # Checking and loading the model weights and its history data from training phase.
    weights_exists = exists(os.readlink(last_model_data_path) + dataset + "_model.h5")
    model_exists = exists(os.readlink(last_model_data_path) + dataset + "_model.json")
    hist_exists = exists(os.readlink(last_model_data_path) + dataset + "_history.json")

if weights_exists and hist_exists:
    print('Existing model saved previously, loading from file...\n')
    with open(os.readlink(last_model_data_path) + dataset + '_history.json', 'r') as infile:
        hist = json.load(infile)
    model.load_weights(os.readlink(last_model_data_path) + dataset + '_model.h5')
    print("Done.\n")
else:
    print("Fitting the model...")
    # The backpropagation algorithm is used in mini-batch mode. Batch size and epochs are set by using
    # default values in Tensorflow example.
    # Cross-validation is performed by using a random split set to 20% of the remaining dataset.
    hist = model.fit(inputs_train, labels_train, batch_size=16, epochs=22, validation_split=val_split,
                     callbacks=utils.trainingCallbacks(log_path=log_path,
                                                       checkpoint_path=model_data_path + dataset + '_model.h5'))
    print("Done.\n")
    # History log data are saved to be used in further graph creation or for experimental use.
    hist = hist.history
    with open(model_data_path + dataset + '_history.json', 'w') as outfile:
        json.dump(hist, outfile)

    if os.path.islink(last_model_data_path):
        os.unlink(last_model_data_path)
        os.symlink(model_data_path, last_model_data_path, target_is_directory=True)
        print("Symbolic Link for Data Model updated.\n")
    else:
        os.symlink(model_data_path, last_model_data_path, target_is_directory=True)
        print("Symbolic Link for Data Model created.\n")

# Re-evaluate the trained model with the same test set.
print("Evaluating Trained Model...\n")
loss, acc = model.evaluate(inputs_test, labels_test, verbose=2)
print("Trained model, accuracy: {:5.2f}% \n".format(100 * acc))

diff = (acc - pre_acc)
if diff >= 0:
    print("Accuracy increased by: {:5.2f}% \n".format(100 * diff))
else:
    print("Accuracy decreased by: {:5.2f}% \n".format(100 * diff))

# Plotting and saving the graphs, respectively, of Loss and Accuracy over Epochs
plt.plot(hist['val_loss'])
plt.title("Val. Loss over Epochs")
plt.ylabel("Val. Loss")
plt.xlabel("Epochs")
plt.savefig(results_path + 'Loss_curve.png', dpi=300)
plt.show()

plt.plot(hist["val_accuracy"])
plt.title("Val. Accuracy over Epochs")
plt.ylabel("Val. Accuracy")
plt.xlabel("Epochs")
plt.savefig(results_path + 'Accuracy_curve.png', dpi=300)
plt.show()

# Use argmax to return the indexes among all the predicted masks with the highest value
prediction = model.predict(inputs_test)
masks_predicted = tf.argmax(prediction, axis=-1)

# Making some random extraction examples of predictions...
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for r in range(3):
    index = np.random.randint(0, len(masks_predicted))
    axes[r][0].imshow(inputs_test[index])
    axes[r][0].set_title("Original")
    axes[r][0].axis('off')
    axes[r][1].imshow(labels_test[index])
    axes[r][1].set_title("Mask")
    axes[r][1].axis('off')
    axes[r][2].imshow(masks_predicted[index])
    axes[r][2].set_title("Prediction")
    axes[r][2].axis('off')
plt.savefig(results_path + 'Predictions.png', dpi=300)
plt.show()
