# Semantic Segmentation [IVU Project A.Y. 2021-2022]

[![CodeFactor](https://www.codefactor.io/repository/github/pilo1996/semantic-segmentation-ivu-project-22-/badge)](https://www.codefactor.io/repository/github/pilo1996/semantic-segmentation-ivu-project-22-)

Semantic Segmentation problem as project for Image &amp; Video Understanding's class in Computer Science @ Ca' Foscari University. Project based on Python, TensorFlow and CityScapes (Pairs) dataset.

## Short Overview
### Labelling the existing dataset
The python script is able to split the images of the dataset and using K-Means it will be able to re-construct the pixel-clusters and predict a label for each of them in order to have a proper labelling ground of truth to be used then in the training phase. The labelling prediction part must be done, for each mask (i.e., labels of the dataset), after the clustering since the aim of this project is to solve the Semantic Segmentation problem. Without labels it is just a Image Segmentation problem. `K-Means` is the clustering method chosen to perform this operation with `K = 10` classes. The following image shows three examples, the original mask provided in the dataset and the labelled mask as raw float data points.

![Example with 3 original examples, with its masks and new labels](/example_org-mask-label.png?raw=true "Example Dataset labels")

### The Neural Network
Segmentation Models library is used to easily instantiate the NN model in just one line of code. The proposed solution is using the [U-Net model](https://arxiv.org/abs/1505.04597) as NN Model. Here some specs:

| What | Value |
|-----------|-----------|
| Model  | `U-Net` |
| Backbone  | `Resnet34` |
| Activation function  | `SoftMax` |
| Optimizer  | `Adam` |
| Learning rate  | `0.01` |
| Loss function  | `Sparse Categorical Cross-Entropy` |
| Metrics  | `Accuracy` |
| Epochs  | `22` |
| Early Stopping (on Acc)  | `max 5 attemps` |

## Dataset
The chosen dataset is the old CityScapes. This dataset format is coming with Pairs Images. We need to split them in half to separate the example space from the output space. In The script the dataset is divided randomly into 3 parts:

| Partition | Nr. of images |
|-----------|-----------|
| Train Set  | `2224` (64%) |
| Validation Set  | `556` (16%)|
| Test Set  | `139` (20%) | 

| Total  | `3475` |
|-----------|-----------|

## Dependencies
- Python 3.9.6
- [Segmentation Models](https://github.com/qubvel/segmentation_models)
- [Tensorflow](https://www.tensorflow.org/?hl=it)
- [Tensorflow Metal Plugin (macOS development)](https://developer.apple.com/metal/tensorflow-plugin/)
- [OpenCV](https://opencv.org)
- [CityScapes Images Pairs](https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs)

## IDE Notes
I have been using miniconda and PyCharm set to the Virtual Environment to get everything working fine. If you want to set up you enviroment in macOS start with the Tensorflow Metal Plugin guide offered by Apple.

## Results
### Predictions
In the following image there are 3 examples with 3 desired outputs and 3 predictions with 22 Epochs. As can be seen, the results are fine but not detailed, especially when it comes to pedestrians segmentation.

![Example with 3 predictions with 22 Epochs](/Predictions.png?raw=true "Three predictions with 22 Epochs")

The following predictions example is made by using a trained model over 100 Epochs

![Example with 3 predictions with 100 Epochs](/Predictions_100.png?raw=true "Three predictions with 100 Epochs")

### Accuracy: `84.12%`
The results are not bad but can be improved. Data Augmentation will help into enhancing the model generalization.

The same model trained over 100 Epochs improved its accuracy estimation up to `86.30%`. 

### Loss Curve 
#### Over 22 Epochs
![Loss Curve with 22 Epochs](/Loss_curve.png?raw=true "Loss Curve with 22 Epochs")

#### Over 100 Epochs
![Loss Curve with 100 Epochs](/Loss_curve_100.png?raw=true "Loss Curve with 100 Epochs")

### Accuracy Curve
#### Over 22 Epochs
![Accuracy Curve with 22 Epochs](/Accuracy_curve.png?raw=true "Accuracy Curve with 22 Epochs")

#### Over 100 Epochs
![Accuracy Curve with 100 Epochs](/Accuracy_curve_100.png?raw=true "Accuracy Curve with 100 Epochs")

### Benchmarks
On 14-inch MacBook Pro (2021, AC powered):
- Based on Apple Silicon M1 Pro (8c CPU, 14c Metal GPU, 16 Neural Engines, 16GB Ram LPDDR5)
- The Execution Time (22 epochs completed) takes ~ `33 Minutes`
- Around 30 Minutes for the Training Time only
- 1 Prediction takes < `79ms` (random image from dataset)
