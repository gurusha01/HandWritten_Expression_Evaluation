# HandWritten_Expression_Evaluation

### This project was done in a team of two : me and [Himanshu Gaurav Singh](https://github.com/cinnabar233).
## Necessary installations 

The following libraries have been used for running the code:
* [`Tensorflow`](https://www.tensorflow.org) for the network architecture
* [`PIL`](https://pypi.org/project/Pillow/) and [cv2](https://pypi.org/project/opencv-python/)or image processing 
* [`Numpy`](https://numpy.org) for computation


## The neural net architecture
The architecture is a CNN(convolutioal network) consisting  of four hidden convolutional layers with ReLU activation and three max-pooling layers followed by a softmax output layer.
The networks is trained over a dataset of digits and the four arithmetic symbols containting around 4800 training images.
Find the dataset [here](https://www.kaggle.com/clarencezhao/handwritten-math-symbol-dataset). 

## Approach

The starting point of the archtecture was the following research [article](https://e-journal.unair.ac.id/JISEBI/article/view/24237) for a CNN over the MNIST dataset. However, due to the smaller size of our dataset, considerable overfitting was observed when the model in the abovementioned article was implemented. This issue was resolved by reducing the number of hidden layers and increasing the extent of maxpooling, in an attempt to decrease the complexity of the model.  
We attempted data augmentation using `augly` for improving model accuracy. However, RAM limits did not permit us to do so.  

## How to run code
```
python3 inference1.py testing-data-path
```
generates a csv file containing the output of the model for the first subtask(classification as infix, prefix or postfix).

```
python3 inference2.py testing-data-path
```
generates a csv file containing the output of the model for the first subtask(value of the arithmetic expression).

The  file `final.h5` is the final learning model.
{"mode":"full","isActive":false}
