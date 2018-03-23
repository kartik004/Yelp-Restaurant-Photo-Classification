# Yelp Restaurant Photo Classification
Deep Learning

## Dataset Link:
[Kaggle Yelp Restaurant photo classification ](https://www.kaggle.com/c/yelp-restaurant-photo-classification/data)


## Description:-
*The Yelp Restaurant Photo Classification challenge is a Kaggle challenge that focuses on the problem predicting user labels of restaurants based on user review photographs. This project approached the problem with the Sequential model convolutional neural network architecture and a custom ensemble approach to achieve a labels for images*.

The goal of the Yelp restaurant photo classification challenge [1] is to build a model that automatically tags restaurants with multiple labels using a dataset of user submitted photos. Currently, Yelp users manually select restaurant labels when they submit a review. Selecting the labels are optional, leaving some restaurants or only partially-categorized. At Yelp, there are lots of photos and lots of users uploading photos. These photos provide rich local business information across categories. Teaching a computer to understand the context of these photos is not an easy task. Yelp engineers work on deep learning image classification projects in-house. 

These labels are:
```
0: good_for_lunch
1: good_for_dinner
2: takes_reservations
3: outdoor_seating
4: restaurant_is_expensive
5: has_alcohol
6: has_table_service
7: ambience_is_classy
8: good_for_kids
```
These labels are annotated by the Yelp community. The task is to predict these labels purely from the business photos uploaded by users. 

Since Yelp is a community driven website, there are duplicated images in the dataset. They are mainly due to:

    users accidentally upload the same photo to the same business more than once (e.g., this and this)
    chain businesses which upload the same photo to different branches

Yelp is including these as part of the competition, since these are challenges Yelp researchers face every day. 

To deter hand labeling, Kaggle has supplemented the test set with additional "ignored" businesses. These are not counted in the scoring. 
File descriptions

    train_photos.tgz - photos of the training set
    test_photos.tgz - photos of the test set
    train_photo_to_biz_ids.csv - maps the photo id to business id
    test_photo_to_biz_ids.csv - maps the photo id to business id
    train.csv - main training dataset. Includes the business id's, and their corresponding labels. 
    sample_submission.csv - sample submission and the test dataset. This is the correct format for your predictions. It should include the business_id and the corresponding predicted labels.

 
 
This assignment is divided in three parts:-


**1) Data Preparation which is done in 	YelpTrainModel-Part1.ipynb. Image data is basically changed to vector format and saved                 in yelpdataset12345.h file.**
 
**2) YelpTrainModel-Part2.ipynb file is used for training the data using CNN.**

**3) Testing with new data using TestYelp.ipynb.**



## Steps to execute the files:

1) First we created the file name append.csv its the combination of file  train_photo_to_biz_ids.csv and train .csv, the reason behind this to map photo's with the labels which are in seperate files as you read in file descriptions

2) This step includes to create a h5 file for training set which includes photographs with respective labels in it. File to use for creation is [YelpTrainModel-Part1.ipynb](https://github.com/kartik004/Yelp-Restaurant-Photo-Classification/blob/master/YelpTrainModel-Part1.ipynb)

3) This step is create the actual model i.e. file [YelpTrainModel-Part2.ipynb](https://github.com/kartik004/Yelp-Restaurant-Photo-Classification/blob/master/YelpTrainModel-Part2.ipynb)

4) As a result of executing the 3rdt step a weights file will be created with higher accuracy which needs to be used further or you can use the mine as well link for [weights.11-0.72365.hdf5](https://drive.google.com/file/d/19a3w-DxSfdwy6m0W5g5JLIJ_26b0IrTv/view?usp=sharing)

5) Final step is to run the [TestYelp](https://github.com/kartik004/Yelp-Restaurant-Photo-Classification/blob/master/TestYelp.ipynb)
         

## Conclusion:
With the Convolution neural network, we have able to predict the label for images present in the test dataset, with final hamming loss of 0.43. Furthermore, we can enhance the project by grouping the images as per business id by which we can label the business as per the group of labels.

