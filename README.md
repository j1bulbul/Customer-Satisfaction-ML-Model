# lz6071ephnpByBW8

Problem Statement: 
Background:

We are one of the fastest growing startups in the logistics and delivery domain. We work with several partners and make on-demand delivery to our customers. During the COVID-19 pandemic, we are facing several different challenges and everyday we are trying to address these challenges. 

We thrive on making our customers happy. As a growing startup, with a global expansion strategy we know that we need to make our customers happy and the only way to do that is to measure how happy each customer is. If we can predict what makes our customers happy or unhappy, we can then take necessary actions. 

Getting feedback from customers is not easy either, but we do our best to get constant feedback from our customers. This is a crucial function to improve our operations across all levels. 

We recently did a survey to a select customer cohort. You are presented with a subset of this data. We will be using the remaining data as a private test set.

Data Description:

Y = target attribute (Y) with values indicating 0 (unhappy) and 1 (happy) customers
X1 = my order was delivered on time
X2 = contents of my order was as I expected
X3 = I ordered everything I wanted to order
X4 = I paid a good price for my order 
X5 = I am satisfied with my courier
X6 = the app makes ordering easy for me 

Attributes X1 to X6 indicate the responses for each question and have values from 1 to 5 where the smaller number indicates less and the higher number indicates more towards the answer. 

Download Data:

https://drive.google.com/open?id=1KWE3J0uU_sFIJnZ74Id3FDBcejELI7FD

Goal(s):

Predict if a customer is happy or not based on the answers they give to questions asked.

Success Metrics:

Reach 73% accuracy score or above, or convince us why your solution is superior. We are definitely interested in every solution and insight you can provide us.

Try to submit your working solution as soon as possible. The sooner the better.

Bonus(es):

We are very interested in finding which questions/features are more important when predicting a customer’s happiness. Using a feature selection approach show us understand what is the minimal set of attributes/features that would preserve the most information about the problem while increasing predictability of the data we have. Is there any question that we can remove in our next survey?

Key Points about the Repository:

1. The Customer Satisfaction script is the first one I did where I attempted the EDA (data prep, histogram, correlation etc), tested out a variety of different models, and added fine tuning of hyperparameters. You will additionally find an attempt at cross-validation fold for the random forest model as I found it's testing and validation accuracy to be the best with random forest so I wanted to probe this further but my insight could be wrong.

2. DECISION_TREE_Feature_Engineering: This was an attempt to add feature engineering (two attempts made: combine survey responses 1-5 into 0-2 and 3-5 to see if that improved accuracy), added a column that was x_5*x_6 to see if that correlation was strong with Y(happy/unhappy), I dont fully understand how this is important/could benefit accuracy but I tried it anyway. not much success

3. Decision_tree_finetune_more: This was an attempt to add even more fine-tuning to the hyperparameters in a decision tree model, I dont think had a huge impact.

When running these scripts, aside from some of the EDA bits and some extra things, you will see the accuracy score for: training set base model, validation set base model,  training set optimized model, validation set optimized model for all the models tested.
