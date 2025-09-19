#### Introdution
The purpose of this project is to implement two machine learning algorithms and evaluate them using a real-world dataset. The two algorithms in question are the Decision Tree and Random Forest algorithms.

Later, we will also implement the Permutation Importance algorithm and use it to evaluate the Random Forest algorithm.

### Data Analysis
When implementing the Decision Tree algorithm, our hyperparameters are maximum tree depth, maximum features, and the splitting criterion, where we test entropy and gini.

Our dataset is a collection of letters (written as numbers) with associated measurements of average pixel positions and variance pertaining to each character.

We split the dataset into features (x), and lables (y). The features contain the measurements, whereas the lables represent the actual letter. The core of this assignment is to feed our learning algorithms the features and trian them to accurately predict the correct lables. Tuning each hyperparameter and argument to maximize model accuracy.

For the Decision Tree algorithm. Maximum depth governs how many times the tree can split. if the maximum depth is reached, a leaf node is returned with the most common lable. In the default setup, the maximum depth is none, meaning the tree will grow until the data set is exhausted.

max_features is the number of features to consider when evaluating the best split. We have chosen to set the max_features as the square root of the total number of features, as this is recommended in the assignment. Only choosing a small subset of features to consider in each split helps to prevent overfitting.

Entropy and gini are both impurity measures that determine where it is most optimal to splt for maximum information gain (split criterion). We will evaluate which method is more optimal, choosing the criterion argument that yields a higher accuracy.

For the Random Forest algorithm, our hyperparameters are n_estimators, max_depth, the criterion, and max_features.

n_estimators are the number of decision trees in the forest
maximum depth, maximum features, and the criterion argument are the same parameters as with the decision tree


### Data Processing
The dataset test/train ratio is 80/20, which is the standard.

#### Model Selection and Evaluation 
For the Decision Tree algorithm, we tested entropy and gini as our criterion arguments. For max_depth, we tested the default value of none, as well as the values 5, 10, and 20. max_features was tested with the values none, sqrt, and log2.

For the Random Forest, we tested with n_estimators set to 10, 20, and 40. For max_depth and max_features, we reduced the breadth of the tests, only testing none, 5, 10, and sqrt, log2 respectively.

For our model selection method, we chose k-fold cross-validation, as decision trees are prone to overfitting, which this method helps to combat. We used k=5 for our cross-validation.

## Decision Trees
We found through our grid search that the most accurate hyperparameter values were to use gini as our criterion argument, max_depth = 20, and max_features = none. with this set of hyperparameters, we arrived at an accuracy score of 0.8975


## Random Forest 
For the Random Forest, n_estimators = 40, max_depth = none, criterion = entropy, and max_features = sqrt yielded an accuracy score of 0.965. This was achieved with a running time of 4m 53s.

## Further Model Selection Results
Having found our desired hyperparameter values, we then train the model with the full training set, and evaluate it on the test sets. This step yielded a simmilar accuracy score for the Decison Tree, and a slightly improved accuracy score for the Random Forest model.

Decison Tree result = 0.895
Random Forest result = 0.9775



## Comparing DT and RF



## Comparing vs existing implementaions 

### Feature importance

#### Conclusion 