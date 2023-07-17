# lz6071ephnpByBW8

1. The Customer Satisfaction script is the first one I did where I attempted the EDA (data prep, histogram, correlation etc), tested out a variety of different models, and added fine tuning of hyperparameters. You will additionally find an attempt at cross-validation fold for the random forest model as I found it's testing and validation accuracy to be the best with random forest so I wanted to probe this further but my insight could be wrong.

2. DECISION_TREE_Feature_Engineering: This was an attempt to add feature engineering (two attempts made: combine survey responses 1-5 into 0-2 and 3-5 to see if that improved accuracy), added a column that was x_5*x_6 to see if that correlation was strong with Y(happy/unhappy), I dont fully understand how this is important/could benefit accuracy but I tried it anyway. not much success

3. Decision_tree_finetune_more: This was an attempt to add even more fine-tuning to the hyperparameters in a decision tree model, I dont think had a huge impact.

When running these scripts, aside from some of the EDA bits and some extra things, you will see the accuracy score for: training set base model, validation set base model,  training set optimized model, validation set optimized model for all the models I tried. Addiitonally, you will 
