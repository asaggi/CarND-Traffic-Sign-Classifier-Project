#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:  
* Load the data set (see below for links to the project data set).  
* Explore, summarize and visualize the data set.  
* Design, train and test a model architecture. 
* Use the model to make predictions on new images.  
* Analyze the softmax probabilities of the new images.   
* Summarize the results with a written report.  


[//]: # (Image References)

[image1]: ./output/exploratory.png  "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test/183823632.jpg "Traffic Sign 1"
[image5]: ./test/453032175.jpg "Traffic Sign 2"
[image6]: ./test/download.jpeg "Traffic Sign 3"
[image7]: ./test/stock.jpeg "Traffic Sign 4"
[image8]: ./test/stop.jpeg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/asaggi/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 6960
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed among the labels. Which looks like it's not distributed proportionately across all labels.

![alt text][image1]

###Design and Test a Model Architecture

####1. Preprocessing the image data.
- Preprocessing was done using normalization as tought in class. I chose normalization becasue ff we didn't scale our input training vectors, the ranges of our distributions of feature values would likely be different for each feature, and thus the learning rate would cause corrections in each dimension that would differ (proportionally speaking) from one another. We might be over compensating a correction in one weight dimension while undercompensating in another.

This is non-ideal as we might find ourselves in a oscillating (unable to center onto a better maxima in cost(weights) space) state or in a slow moving (traveling too slow to get to a better maxima) state.

It is of course possible to have a per-weight learning rate, but it's yet more hyperparameters to introduce into an already complicated network that we'd also have to optimize to find. Generally learning rates are scalars.

Thus we try to normalize images before using them as input into NN (or any gradient based) algorithm.

Here is an example of an original image after normalization:

![alt text][image2]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution Layer1     	| 1x1 stride, same padding, outputs 28x28x6	|
| RELU					|
| Dropout					|		keep probability of 0.5
| Pooling					|		IP = 28x28x16. OP = 14x14x6									|
| Convolution Layer2     	|  IP = 14x14x6, OP = 10x10x16	|
| RELU					|
| Dropout					|		keep probability of 0.5
| Pooling					|		IP = 10x10x16, OP = 5x5x16									|
| Flattening					|		IP = 5x5x16, OP = 400									|	
| Fully Connected     	|  IP = 400, OP = 120	|
| RELU					|
| Dropout					|		keep probability of 0.5
| Fully Connected     	|  IP = 120, OP = 84	|
| RELU					|
| Dropout					|		keep probability of 0.5
| Fully Connected     	|  IP = 84, OP = 43
								 


####3. Training Model approach

I utilized the AdamOptimizer from within TensorFLow to optimize. Also, I tried a few different batch sizes (see below), but settled at 128 as that seemed to perform better than batch sizes larger or smaller than that. I ran 20 epochs, primarily as a result of time and further performance gains, as it was already arriving at nearly 97-98% validation accuracy, and further epochs resulted in only marginal gains while continuing to increase time incurred in training. Additionally, there is no guarantee that further improvement in validation accuracy does anything other than just overfit the data (although adding dropout to the model does help in that regard).

For the model hyperparameters, I stuck with a mean of 0 and standard deviation/sigma of 0.1. An important aspect of the model is trying to keep a mean of 0 and equal variance, so these hyperparameters attempt to follow this philosophy. I tried a few other standard deviations but found a smaller one did not really help, while a larger one vastly increased the training time necessary.

BATCH_SIZE = 128
EPOCH 20 ...
Validation Accuracy = 0.941

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:   
* training set accuracy of 0.891.   
* validation set accuracy of 0.94.  
* test set accuracy of 0.89. 

I picked Lenet architecture for this problem, as it is a well studied, documented and tested approach for image classification problem at hand. Initially, I tested with various "epoch", "batch size" and "learning rate" values. I noticed that as my epoch's increase, my accuracy started dicreasing. Then I introduced the concept of "dropouts" in the model, which eventually helped to reduce overfitting.   
After that, reducing the "epoch's" didn't seem to reduce the efficiency of my model, but it definitely reduced the run time. So I settled with a value of 20, which was giving me a run time of ~5 minutes on a mac book pro with 16 Gigs of ram and Intel i5 Processor with integrated graphics.  
Changing the batch size from 150 to 128 increased some percentage of validation accuracy. It did increase the run time, but didn't effect the accuracy much. that's why I chose 128 as batch size.  
Droupout's "keep probability" was set to 0.5, so as to keep atleast half test data while training.
I didn't fiddle much with the "learning rate" as it was initially set to 0.001, so it kind of gave me reasonable results in somewhat reasonable time.  

 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

One difficulty may be due to the low resolution of the images.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Railroad     			| Priority road 										|
| Yield					| Yield											|
| 30 km/h	      		| Road work					 				|
| Slippery Road			| Slippery road      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. The original model was giving test set accuracy of 89%, which was way above the the accuracy on random images pulled from the net.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign, and the image does contain a stop sign. The top five soft max probabilities were  7.53496230e-01,   2.44850814e-01,   1.28227135e-03, 1.31779772e-04, 7.23789635e-05 and and predicted classes were [17, 14, 15, 13, 22], the highest probability was of "No Entry" sign, which is correct.

For the second image, the top five soft max probabilities were  2.32064888e-01,   1.46351159e-01,   1.36603251e-01, 1.30658001e-01, 1.18910760e-01], and and predicted classes were [23, 11, 19, 30, 12], the highest probability was of "Slippery road" sign, which is correct.

For the third image, the top five soft max probabilities were  7.76498497e-01,   1.71019733e-01,   2.48671062e-02, 9.30203497e-03, 4.53942874e-03 and and predicted classes were [13, 10, 12,  9,  5], the highest probability was of "Yield" sign, which is correct.

For the fourth image, the top five soft max probabilities were  9.16136742e-01,   2.31424551e-02,   1.85112618e-02, 1.63072962e-02,   6.41634129e-03 and and predicted classes were [25, 31,  5,  3, 13], the highest probability was of "Road work" sign, which is not correct.

For the fifth image, the top five soft max probabilities were  9.99998569e-01,   5.80454355e-07,   3.76286920e-07, 1.61356624e-07,   9.77846071e-08 and and predicted classes were [12, 41, 17, 10, 32], the highest probability was of "Priority road" sign, which is not correct.


