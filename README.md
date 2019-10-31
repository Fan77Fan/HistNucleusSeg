# HistNucleusSeg

### Introduction
In this project we are trying to develop an algorithm for segmentation 
nucleus in 2D RGB histology images. 
Due to the limit on the computational power I have 
(running on my laptop)
as well as on the amount of data 
(11 sets of images),
I am using random forest as my
basic statistical model for the task. 
The problem thus is framed as a pixel-wise classification task,
trying to differentiate whether a pixel is a nuclei or not, 
given a set of features on that pixel.

**Eq.1: y = f(X)**

The different methods we explore in this study
focus on constructing different sets of 
features **X**'s that will be helpful for such classification task. 
I use five-fold cross-validation for evaluating different methods,
measuring model classification performance using AUC of ROC.

### Single-level Baseline methods
The baseline method is to use the original values 
in the image RBG channels as input features, 
denoted by $X_0$. 

Classification results using $X_0$ as input features are:

| | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean (std)|
|---|---|---|---|---|---|---|
| baseline | NaN | NaN | NaN | NaN | NaN | NaN |

#### super pixel features
I then add features computed using super pixels to the classifier.
Super pixel is a cluster of neighboring pixels 
in a digital image 
that share the similar visual patterns 
(colour or brightness).
They are usuallly 
calculated from unsupervised machine learning methods. 
Here I use the method from Felzenszwalb for calculating such
super pixels. 
A visualization of what super pixel is like and a comparison
of different methods [can be found here].

[can be found here]: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html

Three features are calculated from
the super pixels calculated from a raw image at every pixel: 
the average intensity of the gray-level image inside the super pixel,
the standard deviation of the gray-level image inside the super pixel,
and the size of the super pixel. I denote this set of features 
as $X_sp$

Classification results using $X_0 + X_sp$ as input features are:

| | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean (std)|
|---|---|---|---|---|---|---|
| +superpixel | NaN | NaN | NaN | NaN | NaN | NaN |

#### low level image filtering features
I also compile a set of low level image features computed through
simple filterings, including gradients in x and y direction, 
as well as a Laplacian filter. 
I denote this set of features as $X_fi$

Classification results using $X_0 + X_sp + X_fi$ as input features are:

| | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean (std)|
|---|---|---|---|---|---|---|
| +filtering | NaN | NaN | NaN | NaN | NaN | NaN |


### Cascaded Auto-context Classifiers



#### 1) Let’s assume that you want to create a system to outline the boundary of the blood pool (i-contours), and you already know the outer border of the heart muscle (o-contours). Compare the differences in pixel intensities inside the blood pool (inside the i-contour) to those inside the heart muscle (between the i-contours and o-contours); could you use a simple thresholding scheme to automatically create the i-contours, given the o-contours? Why or why not? Show figures that help justify your answer.

First, we design a metric called error percentage to measure how accurate the i-contour estimation is. It is defined as the number of misclassification pixels over total number of pixels inside the region of interest, namely o-contour mask.

**Eq.1: error_pct = N_error / N_total **

Given an arbitrary threshold T to classify myocardium and blood pool for each image, we can calculate the error percentage associated with that threshold T. Hence, given an image and knowing the ground truth classification of the two classes, we can calculate a curve error_pct(T) as a function of T, and calculate the minimal error percentage. This is a theoretical lower bound of error percentage, namely, the best possible solution we can get using a single thresholding scheme.

**Eq.2: error_pct_best = min( error_pct(T) )**

To give an example (__experiment_threshold.py__), we select the data pair #2 (SCD0000201, SC-HF-I-2) at slice 120. This is an illustration of images and masks for myocardium and blood pool:
  ![Alt text](segs/model/example_masks_gt.png?raw=true "Title")

This is an illustration of histograms of the two classes and the error percentage curve as a function of threshold T. The best possible error percentage is 18% in this case.
  ![Alt text](segs/model/experiment_hist_thresh_error.png?raw=true "Title")

We also tried a simple thresholding scheme using Otsu's method (https://en.wikipedia.org/wiki/Otsu%27s_method), implemented using toolkit **skimage.filters.threshold_otsu**. The method finds a threshold that minimizes a weighted sum of intra-class variance. As in this example, the error percentage is 29%. A result of this method is illustrated here:
  ![Alt text](segs/model/example_masks_otsu.png?raw=true "Title")

For a final comparison (__main_threshold.py__), we computed the error percentage in all images, both the theoretical lower bound __error_pct_best__ and the error percentage from the Otsu method. The result is 
```
mean best error percentage is 0.120835845404
mean error percentage using Otsu algorithm is 0.345768146028
```

In conclusion, we think using a thresholding scheme to separate myocardium and blood pool is possible, as shown by the results that the best error percentage is around 12%. However, the implementation needs further deliberation. Ostu method, a simple algorithm available in open-source toolkits, may not generate satisfying results.

