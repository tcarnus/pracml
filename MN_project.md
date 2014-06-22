---
title: "pracML"
output: html_document
author: "Tim Carnus"
---


Practical Machine Learning Project - Weight Lifting
========================================================

## Introduction

This report outlines the development of a model which will be used to predict the way that a weight lifting exercise is performed based on personal sensor data.


### Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, we use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The goal of this project is to predict the manner in which a group of 6 participants performed the exercise, captured by the `classe` variable in the training set. 

This report describes:
 - data
 - the model building process
 - the model validation process 
 - a short discussion


```r
#modelling
library(caret)
library(pROC)
```

## Data 
The data for this project come from the following source: (http://groupware.les.inf.puc-rio.br/har). 
The training and test data for this project were directly downloaded from the following locations:


```r
wtrain <- read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', header = TRUE, na.strings=c("NA", ""))
wtest <- read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', header = TRUE, na.strings=c("NA", ""))
```

### Tidy data set
Lets look at the data set:



We note a number of variables unsuitable for predictive model as they stand, including a number of large number of missing values, metadata variables (id, username) and some near-zero-variance predictors. Let us remove them: 

- Large numbers of missing values are not useful for predictor vairables, therefore we remve al variables that have one or more issing values.
- Some of the variables are metadata (ie they give information on the subjects, the structure of the data and the sequence of events). These variable do not contain useful information for predicting the target variable.



```r
# discard NAs
NAfeats <- sapply(wtrain, function(x) {sum(is.na(x))}) 
wtrain <- wtrain[,which(NAfeats == 0)]
# remove metadata variables
INDmeta <- grep("timestamp|X|user_name|new_window|num_window", names(wtrain))
wtrain <- wtrain[,-INDmeta]
```

### Correlations
Correlation amongst predictors can be an issue for some machine learning algorithms and could lead to subptimal model fitting and predictions. Lets look at the correlation between predictor variables so that we may get an understanding of the data we are dealing with.


```r
datype <- sapply(wtrain, class)
numcor <- cor(wtrain[,datype == 'numeric'], use = 'pairwise.complete.obs')
highCorr <- findCorrelation(numcor, 0.80)
numcor[highCorr,highCorr]
```

```
##                   yaw_belt gyros_forearm_z gyros_dumbbell_x gyros_arm_y
## yaw_belt          1.000000        0.073252         0.001599   -0.215525
## gyros_forearm_z   0.073252        1.000000        -0.914476   -0.008836
## gyros_dumbbell_x  0.001599       -0.914476         1.000000    0.015733
## gyros_arm_y      -0.215525       -0.008836         0.015733    1.000000
```

We only identify 2 variables with strong correlation between them. This is unlikely to cause an issue and so we keep all data.


### Training and testing sets

We take a 60% training set to train our model and keep a 40% set for assesing out of sample predictive accuracy.


```r
set.seed(123)
# make training set
INDtrain <- createDataPartition(y = wtrain$classe, p = 0.6, list=FALSE)
Dtrain <- wtrain[INDtrain,]
Dtest <- wtrain[-INDtrain,]
```


# Model building

We use the Caret package for building a predictive model for the `classe` variable from sensor data. For all models we use k-fold cross-validation to tune the model, setting `k = 4` as a sensible value allowing for computation speed and accuracy.

We will look at three modelling approaches, beginning with a simple decision tree, fit using the CART algorithm.


```r
dt1 <- train(classe ~ ., data = Dtrain, method = 'rpart',
             trControl = trainControl(method = 'cv', number = 4))
```

Testing the accurrcay of this mode on the testing set shows that this model does not give us very good acuracy:


```r
testPred <- predict(dt1, Dtest)
dt_acc <- postResample(testPred, Dtest$classe)
```

Accuracy for a simple decision tree is 50.7 .

A slightly more involved algorithm for a classification predicitve model is the Support Vector Machine:


```r
svm1 <- train(classe ~ ., data = Dtrain, 
             method = 'svmRadial',
             trControl = trainControl(method = "cv", number = 4),
             allowParallel=T)
```

Does this model provide better accuracy than the simple decision tree? testing how this model performs is straightforward, predicting the outcomes on the test data set:


```r
testPredsvm <- predict(svm1, Dtest)
svm_acc <- postResample(testPredsvm, Dtest$classe)
```

The SVM model performs significantly better, with an accuracy of 9.2%.

Can this be improved by using a random forest of trees?


```r
rf1 <- train(classe ~ ., data = Dtrain, 
             method = 'rf',
             trControl = trainControl(method = "cv", number = 4),
             allowParallel=T)
```

Using the test data to look at predictive ability of the model:

```r
testPredrf <- predict(rf1, Dtest)
rf_acc <- postResample(testPredrf, Dtest$classe)
```

shows that accuracy is further improved by 6.8% to 99.2%.


## Out of sample error
The out of sample error is calculated simply by substracting the cross-validation derived accuracy value from 1:
We expect out of sample error for the random forest model to be 0.83%.

# Test cases
We now use our best model to predict the class for 20 different cases.

```r
predict(rf1, wtest)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

