#### Introdution
The purpose of this project is to implement two machine learning algorithms and evaluate them using a real-world dataset. The two algorithms in question are the Decision Tree and Random Forest algorithms.

Later, we will also implement the Permutation Importance algorithm and use it to evaluate the Random Forest algorithm.

### Group Members
-Marius Reikerås
-Lyder Samnøy

### Data Analysis
When implementing the Decision Tree algorithm, our hyperparameters are maximum tree depth, maximum features, and the splitting criterion, where we test entropy and gini.

Our dataset is a collection of letters (written as numbers) with associated measurements of average pixel positions and variance pertaining to each character.

We split the dataset into features (x), and lables (y). The features contain the measurements, whereas the lables represent the actual letter. The core of this assignment is to feed our learning algorithms the features and trian them to accurately predict the correct lables. Tuning each hyperparameter and argument to maximize model accuracy.

For the Decision Tree algorithm. Maximum depth governs how many times the tree can split. if the maximum depth is reached, a leaf node is returned with the most common lable. In the default setup, the maximum depth is none, meaning the tree will grow until the data set is exhausted.

max_features is the number of features to consider when evaluating the best split. We have chosen to set the max_features as the square root of the total number of features, as this is recommended in the assignment. Only choosing a small subset of features to consider in each split helps to prevent overfitting.

Entropy and gini are both impurity measures that determine where it is most optimal to splt for maximum information gain (split criterion). We will evaluate which method is more optimal, choosing the criterion argument that yields a higher accuracy.

For the Random Forest algorithm, our hyperparameters are n_estimators, max_depth, the criterion, and max_features.

n_estimators are the number of decision trees in the forest
maximum depth, maximum features, and the criterion argument are the same parameters as with the Decision Tree


### Data Processing
The dataset test/train ratio is 80/20, which is the standard. We also use k-fold cross validation and a random seed.

#### Model Selection and Evaluation 
Preformance tests for the algorithms were mainly done through grid search to find the hyperparameter values which yiled the highest accuracy. For the Decision Tree algorithm, we tested entropy and gini as our criterion arguments. For max_depth, we tested the default value of none, as well as the values 5, 10, 15, and 20. max_features was tested with the values none, sqrt, and log2.

For the Random Forest, we tested with n_estimators set to 10, 20, 30, and 40. For max_features, we reduced the breadth of the tests, only testing sqrt and log2.

For our model selection method, we chose k-fold cross-validation, as decision trees are prone to overfitting, which this method helps to combat. We used k=5 for our cross-validation.

## Decision Trees
We found through our grid search that the most accurate hyperparameter values were to use gini as our criterion argument, max_depth = 10, and max_features = none. with this set of hyperparameters, we arrived at an accuracy score of 0.8975

This however, is not an optimal set of parameters for a decision tree, as only choosing the parameters that yield the highest accuracy leads to preformance loss and overfitting.

When comparing the training data and the test data, we found that accuracy scores diverge with max_depth between 5 and 10. We therefore conclude that max_depth = 7 is optimal for our model, and that anything above that likely leads to overfitting. We also decided to set max_features to sqrt instead of none to prevent overfitting and increase efficiency. When max_features is set to none, every split is considered, which often leads to very simmilar trees that split at around the same dominant feature. To create a robust set of Decision Trees for use in the Random Forest, it is important to only consider a small subset of features each time, so the trees are different enough for ensemble learning to be possible.

With these changes, and using the full training set, accuracy decreases to 0.7678. However, this model should not overfit or split at the same values each time.


## Random Forest 
For the Random Forest, n_estimators = 40, max_depth = none, criterion = entropy, and max_features = sqrt yielded an accuracy score of 0.965. Again, we need to tweak the parameters to achieve a more robust model. Following the exaple in our Decision Tree, and after analyzing the results graphically, we set n_estimators = 30, max_depth = 5, criterion = gini, and max_features = sqrt. On the full training data, this reduces the accuracy score to 0.8925.

## Comparing DT and RF
When developing our Decision Tree algorithm, we found that to prevent overfitting, the trees needed to be small enough that high variance in the accuracy rating per tree was inevitable. This is a flaw that the Random Forest algorithm is designed to counteract. With the Random Forest, averaging several trees helps to combat the high variance of each tree, leading to a more accurate average. The Random Forest also reduces overfitting by averaging over a diverse set of trees. This illustrates how ensemble learning helps to create more robust and accurate model, at the cost of a higher runtime.



## Comparing vs existing implementaions 
When testing the SKlearn algorithms with the same hyperparameters, we arrived at very simmilar accuracy scores:

Our Decision Tree: 0.7675
Our Random Forest: 0.8875

SKlearn Decision Tree: 0.7625
SKlearn Random Forest: 0.8775

### Feature importance

#### Conclusion 