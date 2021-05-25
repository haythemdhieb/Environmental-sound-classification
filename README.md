# Environmental sound classification
This project aims to classify environmental sounds using two supervised machine learning algorithms: KNN and SVM
### First step: feature extraction and dataset creation
In this step, we use a lot of audio descriptors to extract features from audio data. For this task, we use different types of audio descriptors: temporal (zero corssing rate), spectral(spectrogramme and spectral flatness) and cespstral (MFCC: mel frequencey cesptral coefficient).
NB: Different from traditional methods that uses deep neural networks for image classification extracted using MFCC, we use a lot audio descriptors and combine together  and test the result. 
### Second Step: training
After extracting the relevant features, we reduce the number of features to 10 (from 29 to 10) to reduce the dimensionality of the problem, then, we use KNN and then SVM to for testing. As a result, around 80% of accuracy for both algorithms on data sets of 10 different classes.
