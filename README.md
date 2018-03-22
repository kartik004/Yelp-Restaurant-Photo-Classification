# Yelp Restaurant Photo Classification
Deep Learning

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


** 1) Data Preparation which is done in 	YelpTrainModel-Part1.ipynb. Image data is basically changed to vector format and saved                 in yelpdataset12345.h file.**
 
** 2) YelpTrainModel-Part2.ipynb file is used for training the data using CNN.**

 3) Testing with new data using TestYelp.ipynb.

## Steps to execute the files


         



