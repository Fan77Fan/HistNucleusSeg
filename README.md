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
features $X$'s that will be helpful for such classification task. 
I use five-fold cross-validation for evaluating different methods,
measuring model classification performance using **AUC of ROC**.
Because of the fact that very little hyper-parameter tuning 
is needed (choice of random forest), I do not have validation
set in this study (i.e. only a split of training/test set in
each fold).

### Single-level Baseline methods
The baseline method is to use the original values 
in the image RBG channels as input features, 
denoted by $X_0$. 

Classification results using $X_0$ as input features are:

| | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean (std)|
|---|---|---|---|---|---|---|
| baseline | 0.885 | 0.931| 0.960 | 0.951 | 0.885 | 0.923 (0.032) |

A demonstration from one of the test images is shown here:
  ![Alt text](./result/demo_baseline.png?raw=true "Title")

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
Here we denote the super pixel image calculated from the
raw RGB image as $I_sp$

[can be found here]: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html

Three features are calculated from
the super pixels calculated from a raw image at every pixel: 
the average intensity of the gray-level image inside the super pixel,
the standard deviation of the gray-level image inside the super pixel,
and the size of the super pixel. I denote this set of features 
as $X_sp$

Classification results using $X_0 + X_sp$ as input features are
(here the $+$ sign stands for concatenation):

| | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean (std)|
|---|---|---|---|---|---|---|
| +superpixel | 0.890 | 0.935 | 0.961 | 0.962 | 0.898 | 0.929 (0.030) |

A demonstration from one of the test images is shown here:
  ![Alt text](./result/demo_superpixel.png?raw=true "Title")


#### low level image filtering features
I also compile a set of low level image features computed through
simple filterings, including gradients in x and y direction, 
as well as a Laplacian filter. 
I denote this set of features as $X_fi$

Classification results using $X_0 + X_sp + X_fi$ as input features are
(here the $+$ sign stands for concatenation):

| | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean (std)|
|---|---|---|---|---|---|---|
| +filtering | 0.891 | 0.939 | 0.962 | 0.964 | 0.899 | 0.931 (0.031) |

A demonstration from one of the test images is shown here:
  ![Alt text](./result/demo_filtered.png?raw=true "Title")

Note that in these baseline methods, the classification is
performed using only features from the single pixel, 
with the exception of the super pixel features where
neighbourhood information is included. Next we are going
to explore ways of integration neighbourhood information
to improve the predictive power.

### Cascaded Auto-context Classifiers

The idea comes from the [works in auto-context], 
which is to train a cascade of classifiers, 
each taking outputs generated from the previous classifier
as additional input features,
in order to boost classification performance.
The whole classifier, which consisted of a cascade of 
individual classifiers, then is trained in a multi-phase manner.
I denote the input and output of each classifier as $X_i$ 
and $Y_i$, respectively. In the first phase of training,
$X_i$ is the same as the full set of image features described
in the previous section $X_0 + X_sp + X_fi$. 

In this project, I further expand the set of features for input
in each phase of training by adding: 
the predicted scores from the 
previous classifier **in a neighbourhood** (${Y_i}$), 
and the gray-scale image intensity values **in a neighbourhood**.
The scope pf neighbourhood is different in different phases
of training. 
In the second phase, it is a 3x3 matrix with a 3-pixel length
between nodes. 
In the third phase, it is a 3x3 matrix with a 6-pixel length
between nodes.
I denote this set of feature as auto-context neighbourhood
feature $X_acn_i = G(Y_{i-1})$.

I also add features computed using the predicted score $Y_i$
and the super pixel image $I_sp$ that I calculated in the 
beginning, by calculating the mean and standard deviation of
$Y_i$ inside each super pixel. 
All of these is to include information from the auto-context 
(i.e. classification performance from the previous classifier),
as well as information from the neighbourhood.
I denote this set of feature as auto-context super pixel feature
$X_acsp = H(Y_{i-1}, I_sp)$

[works in auto-context]: https://pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf

In my three phases of training, the classification performance
advances as such:

| | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean (std)|
|---|---|---|---|---|---|---|
| phase 1 | 0.894 | 0.940 | 0.962 |0.964 | 0.900 | 0.932 (0.030) |
| phase 2 | 0.910 | 0.951 | 0.968 | 0.971 | 0.905 | 0.941 (0.028) |
| phase 3 | 0.914 | 0.953 | 0.969 | 0.972 | 0.906 | 0.943 (0.028) |

Note that phase 1 is the same as the last baseline method
when all image-level features ($X_0 + X_sp + X_fi$) 
are included. The numerical difference in results comes
from the randomness in random forest methods.

A demonstration from one of the test images is shown here:
  ![Alt text](./result/demo_phase1.png?raw=true "Title")
  
  ![Alt text](./result/demo_phase2.png?raw=true "Title")
  
  ![Alt text](./result/demo_phase3.png?raw=true "Title")


### Discussion
In this study, we explore different algorithms for pixel-level
classification in 2D RGB histology images. The difference
in algorithms focuses on the set of features used for training
random forest models. We see some improvement of performance
as we include more and more advanced features that take the
advantages of neighbourhood information, cascaded classifier
information, and unsupervised image prior (super pixels). 
This is by no means claiming the classification performance
at the end of the study to be optimal, but simply to 
show a preliminary exploration of constructing/validating/comparing
different methods. 

We observe that the model still makes certain mistakes in the
task. [Some further reading] shows that different subtypes of
cells/nucleus demonstrate different imaging characteristics. 
We can see that the model is better at classifying nucleus that 
have strong intensity values, worse otherwise. 
The high performance variation in between folds suggests that 
our data may have consisted several subtypes of cells.
To solve this in the future, we could include features that 
incoporate some description of shape in a larger scale. 
We could either apply level-set methods on the current 
classification result, and use that as input features in our
cascaded classifiers,
or simply use convolutional neural
network methods.

[Some further reading]:https://www.histology.leeds.ac.uk/cell/nucleus.php

