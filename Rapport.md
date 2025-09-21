mariusrei
mariusrei
Online

mariusrei — Yesterday at 22:46
Men den kode blokken er ikke viktig, den bare gir er info
djhellaswag — Yesterday at 22:46
jeg sjekker filen manuelt og da har den oppgave 4, men når jeg trykket "open with VS code" rett under så hadde den ikke endringene...
mariusrei — Yesterday at 22:47
jeg får bare dette av den

=== Custom DecisionTree ===
              precision    recall  f1-score   support

           0     0.9500    0.8906    0.9194        64
           1     0.8219    0.8696    0.8451        69
           2     0.9688    0.9538    0.9612        65
           3     0.8806    0.8806    0.8806        67
           4     0.9344    0.8906    0.9120        64
           5     0.8667    0.9155    0.8904        71

    accuracy                         0.9000       400
   macro avg     0.9037    0.9001    0.9014       400
weighted avg     0.9020    0.9000    0.9005       400

Confusion matrix:
 [[57  0  2  2  0  3]
 [ 0 60  0  3  4  2]
 [ 0  2 62  0  0  1]
 [ 0  6  0 59  0  2]
 [ 3  1  0  1 57  2]
 [ 0  4  0  2  0 65]]

=== Custom RandomForest ===
              precision    recall  f1-score   support
...
 [ 0  0 64  0  1  0]
 [ 0  0  0 67  0  0]
 [ 0  1  1  0 62  0]
 [ 0  2  0  1  0 68]]
djhellaswag — Yesterday at 22:47
ja samme her
mariusrei — Yesterday at 22:49
Men det går vel ann å fjerne den delen, vi den ikke, jeg bare ville ha mer info for å forstå
men alt det andre fungerer?
djhellaswag — Yesterday at 22:50
ja, men d er ugler i mosen ass
mariusrei — Yesterday at 22:50
forsår ikke hvordan du får error men ikke jeg
men vi trenger den ikke kun sammenligning av accuracy og oppgave 4
djhellaswag — Yesterday at 22:52
bare gjør ctrl shift f og ta et screenshot av alt som kommer opp når du søker på dt_test_pred
mariusrei — Yesterday at 22:54
ser nå, mener det bare skal stå dtpred ikke dt test_pred
djhellaswag — Yesterday at 22:55
dtpred finnes heller ikke for meg
mariusrei — Yesterday at 22:56
ikke her?
Image
djhellaswag — Yesterday at 22:56
jo med underscore ja
den runner da
men vet ikke hva den gjør
mariusrei — Yesterday at 22:57
Vi bare dropper den
djhellaswag — Yesterday at 22:58
ait, da ser jeg på å legge max_depth inn i final evaluation, altså droppe none som hyperparameter, også skriver jeg videre på rapporten
kanskje du kan se på de premade algoritmene vi skulle sammenligne med, se hva de får som accuracy
mariusrei — Yesterday at 23:00
yeye, men tar det imrg, pusher koden nå som jeg fjernet den + det var noe rart med random seed som jeg tror jeg fikset
djhellaswag — 01:32
ok, jeg tror vi må skrive om hvordan vi bruker "best params" rett fra grid search i den ferdige modellen vår

best params fra grid search velger de parameterene som gir best accuracy selv når det går på bekostning av resten av modellen (runtime og overfitting for dt)

selv om best max depth fra grid search i dt'et er 10, ser jeg at train og test scoren divergerer (start på overfitting) når max depth = 5, selv om 10 er mer "accurate". så 5 er nok verdien vi bør bruke

på n_estimators har jeg testet en del, og det ser ut som n = 30 er mest optimalt
mariusrei — 14:11
Har gjort litt småfiking på koden nå, bare sjekk om den runenr for deg også
djhellaswag — 14:12
ja samme her lol, jeg bare sjekker dine commits, ser hva som skjer
har ikke pushet mitt
djhellaswag — 14:27
vet du hvorfor vi får entropy som beste criterion? bør vi bruke gini likevel?
mariusrei — 15:01
Usikker, kan det være pga størrlsen på datasettet? 1600 ?
djhellaswag — 15:02
vet ikke, men gini gir mye bedre grafer
mariusrei — 15:02
hmm...
Hva mener du med bedre grafer?
djhellaswag — 15:11
entropy blir ganske random
her er min versjon av final evaluation med verdiene jeg har komt frem til i stedet for "best params"
Image
dt accuracy hopper veldig mellom 0.69 og 0.79
djhellaswag — 15:19
her la jeg inn for SKlearn, fikk ca. d samme
Image
hadde vi ikke noe random seed opplegg som gjorde at vi skulle få de samme verdiene?
får ganske forskjellig hver gang jeg runner
djhellaswag — 15:50
hvordan tester vi "speed" for SKlearn sine?
mariusrei — 15:52
Kan vel bare legge en time() inn før og etter vi kjører den?
noe slikt kanskje?

start = time.time()
sk_rf.fit(X_train, y_train)
traintime_rf = time.time() - start
med alle 4
djhellaswag — 15:53
sånn vi har d nå tar d 0.1 sek
mariusrei — 15:54
vi skal har randomstate, men ser jeg glemte det for sk 
    random_state=0,
skal være     random_state=RANDOM_Seed
Men hvordan kom du fram til paramterer dine?
bare sjekket du manulet? skal ikke vi bruke gridsearch for det?
djhellaswag — 15:57
grid search gir beste accuracy, men beste accuracy er alltid bare det dypeste og mest tidkrevende
jeg skiver om det i rapporten, jeg kan sende den til deg, gidde ikke commite siden jeg har mye annet drit på min nå
mariusrei — 15:58
okei, noe mer du vil at jeg skal se på? Eventuelt hva mer vi trenger i koden
djhellaswag — 15:58
jeg har ikke sett på oppg4
#### Introdution
The purpose of this project is to implement two machine learning algorithms and evaluate them using a real-world dataset. The two algorithms in question are the Decision Tree and Random Forest algorithms.

Later, we will also implement the Permutation Importance algorithm and use it to evaluate the Random Forest algorithm.

### Data Analysis
Expand
Rapport.md
6 KB
du kan lese om hvordan jeg resonerer for hyperparametrene i decision tree og random forest
﻿
djhellaswag
djhellaswag
 
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

When comparing the training data and the test data, we found that accuracy scores diverge when max_depth goes beyond 5. We therefore conclude that max_depth = 5 is optimal for our model, and that anything above 5 leads to overfitting. We also decided to set max_features to sqrt instead of none to prevent overfitting and increase efficiency.

With these changes, and using the full training set, accuracy decreases to between 0.7 and 0.8. However, this model should not overfit to the same extent.


## Random Forest 
For the Random Forest, n_estimators = 40, max_depth = none, criterion = entropy, and max_features = sqrt yielded an accuracy score of 0.965. Again, we need to tweak the parameters to achieve a more robust model. Following the exaple in our Decision Tree, and after analyzing the results graphically, we set n_estimators = 20, max_depth = 5, criterion = gini, and max_features = sqrt. On the full training data, his reduces the accurecy score to 0.8925.

## Comparing DT and RF
When developing our Decision Tree algorithm, we found that to prevent overfitting, the trees needed to be small enough that high variance in the accuracy rating per tree was inevitable. This is a flaw that the Random Forest algorithm is designed to counteract. With the Random Forest, averaging several trees helps to combat the high variance of each tree, leading to a more accurate average. The Random Forest also reduces overfitting by averaging over a diverse set of trees. This illustrates how ensemble learning helps to create more robust and accurate model, at the cost of a higher runtime.



## Comparing vs existing implementaions 
When testing the SKlearn algorithms with the same hyperparameters, we arrived at very simmilar accuracy scores:

Our Decision Tree: 0.7675
Our Random Forest: 0.8875
SKlearn Decision Tree: 0.7625
SKlearn Random Forest: 0.8775

### Feature importance

![alt text](image.png)


Permutation importance is used to check how much each feature actually matters for the model. You shuffle one feature at a time and see how the accuracy changes. A big drop means the feature is important, while little or no change means it isn’t very useful.

In our results, shown in the graph above, almost all features have little or no impact on the model. The two exceptions are **xy2br** and **yege**, which are both way more impactful on the models performance.

A strength of this method is that it is model agnostic, so it works with any kind of model. It is also easy to understand and explain. However, it has some weaknesses. If features are strongly correlated, the method can underestimate their importance, since shuffling one feature may not matter if another correlated one is still present. It also depends on the test data and can be slow on larger datasets because the model has to be run many times.




#### Conclusion 