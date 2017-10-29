# **Traffic Sign Recognition** 



---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report




## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---




### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 1
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

visualization is included in the html file

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 

Seeing that there was a large difference in the number of examples per label, i decided to even it out by creating more examples. 
I did this by using an image transforming function i found online. It takes an image and rotates it, and shrinking/inlarging it. It also applied a random brightness to the image but i removed that function since some of the images where already so dark that it made them almost black. An example of a generated image can be found in the html file. I generated new examples until i had an equal number of examples for each label.

I then tranfered the images to grayscale to make it easier for the network.

Lastly i normalized all their values to between -1 and 1 so that the network wouldn't have to make drastic ajustments for large numbers.




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 1x1     	| 1x1 stride, same padding, outputs 32x32x1 	|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 32x32x1 	|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 32x32x1 	|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x16 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 14x14x32 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 					|
| flatten		      	| outputs 800				 					|
| Fully connected		| input 800, output 400        					|
| RELU					| 	        									|
| Dropout				|												|
| Fully connected		| input 400, output 200        					|
| RELU					| 	        									|
| Dropout				|												|
| Fully connected		| input 200, output 43        					|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer, it preforms really well and i don't know enough to choose a better one. I had a learning rate of 0.0003, i tried many different numbers and this seemed to be the highest i could put it while not effecting the models accuracy negatively. My batch size was 64, this for me was the right balance of performance vs time. Setting it at 64 gave me pretty good results while still maintaining a reasonable training time. i picked 45 epochs after training it many times and seeing when the learning curve started to flatten out.



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 97.2%
* validation set accuracy of 94.8% 
* test set accuracy of 92.7%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

first i tried the LeNet archetecture. 

* What were some problems with the initial architecture?

It worked fairly well and i got around 90%. But that's not quite enough and so i moved on to deeper networks.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

At first i tried simply adjusting the size of the LenNet architecture. making the two convolutional layers deeper, and the fully connected layers bigger, but that didn't work. That's when i started adding layers. I added two extra convolutional layers with no activation function right before the two i already had. I'd seen this type of architecture before for image recognition so i figured it might help here. I also read an article about using 1x1 convolutional layers at the start to adjust the image color range to whatever the nueral net thought was the best, so i did that. Now i had 3 color shifting conv layers in the begginning, and two sets of convolutional layers after that. The network was getting big though so i added dropout to prevent overfitting and made the fully connected layers big enough to handle all the data they were getting passed.


* Which parameters were tuned? How were they adjusted and why?
I significantly lowered the learning rate because it helped the more precisely fit the data.


 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](https://i.imgur.com/vxkPKXQ.jpg?2) ![alt text](https://i.imgur.com/Bs5yV8S.jpg?1) ![alt text](https://i.imgur.com/0Ma9Mfd.jpg?1) 
![alt text](https://i.imgur.com/EOMjH01.jpg?1) ![alt text](https://i.imgur.com/8EfF5e5.png?1)

they all actually seem pretty generic. I don't think any of them will be harder to classify than any of their counterparts in the dataset. Unless they are too well lit...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 30km      		| Speed limit30 km  									| 
| Double Curve    			| Right-of-Way at next intersection 										|
| general Caution					| General Caution											|
| No Entry	      		| No Entry					 				|
| Stop			| Stop      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares to the 97-94% that the model got. This is a small sample but the images are generally nicer that the images in the database so that could be trowing it off.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

In the first image the model is pretty certain that it is a speed limit sign, and 30km. However upon running it multiple times, it does choose 20km sometimes.


### Image 1:

| Guess			        |     Probability        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h) | 96.588% |
| Speed limit (20km/h) | 2.993% |
| Speed limit (60km/h) | 0.209% |
| Speed limit (50km/h) | 0.109% |
| Speed limit (80km/h) | 0.077% |



The second image is interesting because it's 100% sure that it's a "right-of-way at next intersection" sign. But it isn't. This leads me to question whether it would ever predict the double curve.

### Image 2:

| Guess			        |     Probability        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection | 100.0% |
| Beware of ice/snow | 0.0%  |
| Children crossing | 0.0% |
| Dangerous curve to the right | 0.0%|
| Slippery road | 0.0% |



On the third image the model is once again completely certain. But this time it's "general caution" and it's right. This makes sense as it's a very easy to identify sign and doesn't look that much like any other sign.

### Image 3:

| Guess			        |     Probability        					| 
|:---------------------:|:---------------------------------------------:| 
|General caution | 100.0%|
|Bumpy road | 0.0%|
|Pedestrians | 0.0%|
|Slippery road | 0.0%|
|Traffic signals | 0.0%|




In the fourth image it's correctly fairly certain that it's a "No Entry" sign. However, it does think there's a chance that it's a blue roundabout mandatory sign. This is possibly a negative side effect of converting the images to greyscale, or possible the first 3 color shifting convolutional layers.

### Image 4:

| Guess			        |     Probability        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry|  97.696%|
|Roundabout mandatory | 2.302%|
|Keep right |0.001%|
|Speed limit (50km/h) | 0.0%|
|Speed limit (100km/h) | 0.0%|




On the fifth image it has even more trouble determaining the stop sign from the blue roundabout sign. 

### Image 5:

| Guess			        |     Probability        					| 
|:---------------------:|:---------------------------------------------:| 
|Stop |50.332% |
|Roundabout mandatory | 38.926%|
|No passing | 6.719%|
|Yield | 2.959%|
|Vehicles over 3.5 metric tons prohibited | 0.945%|
