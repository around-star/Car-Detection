## Car-Detection
A HOG and SVM based object detection algorithm

## Histogram of Oriented Gradient
[Histogram of Oriented Gradient](https://www.learnopencv.com/histogram-of-oriented-gradients) is a feature descriptor technique which generates features from images.

The features extracted is used for training in the SVM classifier. The classifier model generated is used for predicting whether a certain region contauns the object or not.

## Files
- ``` set_images.py  ``` : Transforms any number of images out of the total set in ```vehicles```and ```non-vehicles``` into grayscale and creates a dataset by storing in ```Pos_img``` and ```Neg_img``` respectively. The number of each set can be decided by the user.
- ```feature_extract.py ``` : Uses the Histogram of Oriented Gradient Descriptor to create the features out of images.
- ```sliding_windows.py ``` : Returns windows extracted from a given image to perform training/detection.
- ```training.py``` : This performs training of the classifier from the features obtained from ```feature_extract.py``` . The trained classifier is stored as a pickle file (**model.pkl**)
- ``` result.py ``` : This performs detection of cars in any given image of choice using the trained classifier (**model.pkl**)
