## Traffic Sign Recognition

---

**Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram.png "Training Set Classes Distribution"
[image2]: ./examples/classes_examples.png "Classes Examples"
[image3]: ./examples/normalization.png "Before and after normalization"
[image4]: ./examples/bumpy road.jpg "Traffic Sign 1"
[image5]: ./examples/Do-Not-Enter.jpg "Traffic Sign 2"
[image6]: ./examples/limit30.jpg "Traffic Sign 3"
[image7]: ./examples/roundabout.jpg "Traffic Sign 4"
[image8]: ./examples/STOP.jpg "Traffic Sign 5"
[image9]: ./examples/barchart.png "Test Results"
[image10]: ./examples/weights.png "First Convolution Layer Features"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/AldoArriagaOrta/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

## Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The first figure is a histogram showing the number of examples per class in the training set.

![alt text][image1]

The second figure presents examples of every class, extracted from the training set.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it would reduce the input size. However, after some tests I got better performance using images with 3 channels (RGB). The only pre-processing step carried out was a simple normalization (code available in cell 4 of the Ipython notebook)

I believe that the degradation of performance after grayscaling was due to the loss of information and features inherent to the colour channels.

Here is an example of a traffic sign image before and after normalization.

![alt text][image3]
 

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1 Input         		| 32x32x3 RGB image   							| 
| 2 Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x64 	|
| 3 RELU				|												|
| 4 Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| 5 Convolution 3x3	    | 1x1 stride, valid padding, outputs 14x14x128 	|
| 6 RELU				|												|
| 7 Max pooling	      	| 2x2 stride,  outputs 5x5x128					|
| 8 Convolution 3x3	    | 1x1 stride, valid padding, outputs 5x5x128 	|
| 9 RELU				|												|
| 10 Max pooling	    | 2x2 stride,  outputs 2x2x256					|
| 11 Flatten			| Output 1024									|
| 12 Fully connected	| Input = 1024. Output = 400.  					|
| 13 RELU				|												|
| 14 Dropout			| 0.5											|
| 15 Fully connected	| Input = 400. Output = 210. 					|
| 16 RELU				|												|
| 17 Dropout			| 0.5											|
| 18 Fully Connecte		| outputs 43 classes       						|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer, a batch size of 128, 30 epochs, and 0.001 learning rate.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 97%
* test set accuracy of 96%

To begin with, I used the LeNet model from the lectures, increasing the input size to accept RGB images as input and also the number of ouputs to allow for 43 classes.
The accuracy for the validation set did not surpass 90%, whereas the accuracy for the training set reached 100%. The network was clearly overfitting, varying the learning rate, batch size and number of epochs did not help to improve the accuracy.

Subsequently, I tried to pre-process the images by normalizing and grayscaling. The performance still was still under the minimum requirement and the grayscaling was showing worse performance than the colour normalized images.

Since overfitting was still being an issue, dropout layers with a keep probability of 0.5 were added. To further improve the accuracy, I added a third convolution layer and experimented with the sizes of filters and outputs.

After this brief experimentation the accuracy achieved surpassed the minimum requirement. The dropout layers proven extremely important for preventing overfitting. It would be interesting to investigate the addition of dropout in the convolution layers.

Some important design choices are the sizes of filters, learning rate, epochs and batch size, as they need to be tuned not only to achieve the required performance but also to minimize the time needed to train the network

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because there are a number of similar caution signs (red triangle) and due to the size reduction of the image the symbol within the sign could be easily misclassified.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road     		| Bumpy Road   									| 
| No entry     			| No entry 										|
| Speed Limit(30 km/h)	| Speed Limit(30 km/h)							|
| Roundabout mandatory	| Roundabout mandatory			 				|
| STOP					| STOP     										|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 9h cell of the Ipython notebook.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0  			   	      	| Bumpy Road   									| 
| 1.0 		   			| No entry 										|
| 1.0					| Speed Limit(30 km/h)							|
| 1.0					| Roundabout mandatory			 				|
| 1.0					| STOP     										|
|  |  |

![alt text][image1]
![alt text][image9]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Some Features seem to concentrate on the circle formed by the arrows (e.g. Features 0 and 28 ) while some others seem more responsive to the arrows shapes (e.g. Features 7 and 44). However, the interpretation of these images still seem a little obscure.

![alt text][image10]
