---
layout: post
title:  "Semantic Segmentation for Satellite Imagery with fastai"
date:   2020-06-13 17:43:42 +0200
---

Semantic segmentation is a classification task in computer vision that assigns a class to each pixel
of an image, effectively segmenting the image into regions of interest. For example, the
Cambridge-driving Labeled Video Database (CamVid) is a dataset that includes a collection of videos
recorded from the perspective of a driving car, with over 700 frames that have been labeled to
assign one of 32 semantic classes (e.g. pedestrian, bicyclist, road, etc.) to each pixel. This
dataset can be used to train a model to segment new scenes into those same classes.

{:class="image-pad"}
![Example labeled frames from the CamVid dataset](/assets/images/semseg/camvid.png)
*Labeled frames from the CamVid dataset. Each color in the label images corresponds to a
different class.*

One application of semantic segmentation is the labeling of satellite imagery into classes based on
land use. The Inria Aerial Image Labeling Dataset is a collection of aerial images covering several
cities from around the world, ranging from densely populated areas to alpine towns, accompanied by
ground truth data for two semantic classes: "building" and "not building". In other words, the Inria
dataset can be used to train a pixelwise classifier that can identify human-made structures from
satellite images.

{:class="image-pad"}
![Inria - Chicago](/assets/images/semseg/inria-chicago.jpg){:style="width: 40%; margin: 10px"}
![Inria - Chicago label](/assets/images/semseg/inria-chicago-label.jpg){:style="width: 40%; margin: 10px"}  
*Chicago*

{:class="image-pad"}
![Inria - Kitsap County, WA](/assets/images/semseg/inria-kitsap.jpg){:style="width: 40%; margin: 10px"}
![Inria - Kitsap County, WA label](/assets/images/semseg/inria-kitsap-label.jpg){:style="width: 40%; margin: 10px"}  
*Kitsap County, WA*

{:style="text-align: center"}
*Images from the Inria dataset with their corresponding labels.*

Fast.ai covers semantic segmentation for the CamVid dataset in their Practical Deep Learning for
Coders course. After taking the course, I wanted to see if I could apply what I'd learned by using
the Fast.ai library and a pretrained ImageNet classifier to train a model on the Inria dataset with
competive performance. This post documents my work on the project, including some techniques I
learned for working with large format satellite imagery and experimenting with custom model
architectures and the Fast.ai library.

## The Inria Dataset

As described above, the Inria dataset is a collection of satellite images covering 10 different
cities around the world. Each image, or tile, is a 5000 by 5000 pixel color image that covers an
area of 2.25 km<sup>2</sup>, so the width of a single pixel corresponds to 0.3 m. Half of the images
are accompanied by ground truth data, which are single channel images of the same size where pixels
corresponding to the building class have value 255 and all others have value 0. These images are
used for model training and evalatuion, while the unlabeled images are used as a test set. There is
no overlap in cities in the train and test sets; each subset contains 36 images from 5 different
cities for a total of 180 images. The authors of the dataset maintain a leaderboard for performance
over the test set according to the metrics described below. As suggested by the authors, I used the
first tile from each city in the training set for validation.

Given the size of the images relative to available GPU memory, each tile and its label was split up
into patches as a preprocessing step. I chose a patch size of 256 by 256 pixels, which allowed me to
use a batch size of 32 during training. I used reflection padding to enlarge the tiles so that they
could be evenly split into patches. Given the importance of context in identifying building pixels,
I used larger, overlapping patches for the validation and test sets, in a process that is described
in more detail below.

{:class="image-pad"}
![Patches](/assets/images/semseg/inria-patches.png){:style="max-width: 60%"}  
*Patches from one of the tiles in the Inria dataset. Notice the reflection padding in the
edge patches.*

## Metrics

The evaluation metrics used in the Inria competetion are pixelwise accuracy and intersection over
union (IoU) of the building class. IoU is a common evaluation metric in both semantic segmentation
and object detection tasks that is defined as the size of the intersection between the predicted and
ground truth pixel areas or bounding boxes and the size of their union.

{:class="image-pad"}
![Intersection over Union](/assets/images/semseg/iou.png)
*Intersection over Union is defined as the area of overlap between two sets or areas divided by the
area of their union.*

You'll notice from the competition leaderboard that the accuracy values are generally quite high:
all are over 93%. This is partially due to a large class imbalance in the dataset. Approximately 15%
of the pixels in the training set belong to the building class, so a model that predicted "not
building" for all pixels would have an accuracy of around 85%. IoU for the building class, on the
other hand, effectively normalizes for the size of the building class to give a measurement of how
well the model identifies buildings. For that reason, I focused mainly on optimizing IoU over the
validation set.

For training I used cross entropy loss, which is the default for semantic segmentation in fastai. I
also experimented with soft dice loss, an alternative loss function introduced for medical image
segmentation which, like IoU, normalizes for the size of each class in the calculation. I expected
this to give me a better IoU as it seemed to be more directly optimizing the quantity of interest,
but was not able to beat the performance of a model trained with cross entropy loss. I didn't look
into this in too much detail, but I suspect the dynamics of training with soft dice loss were
somehow making it harder for the model to converge to a minimum that would generalize well.

{:class="image-pad"}
$$ 1 - \frac{2\sum_{pixels} y_{true} y_{pred}}{\sum_{pixels} y_{true}^2 + \sum_{pixels} y_{pred}^2} $$  

{:style="text-align: center"}
*Calculation of soft dice loss. The scoring is repeated over all classes and averaged.*
