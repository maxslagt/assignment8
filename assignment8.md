---
title: "Practical Machine Learning Project"
author: "Max Slagt"
date: "June 25, 2019"
output:
  html_document:
    keep_md: yes
  pdf_document: default
  word_document: default
---

# Executive Summary
#### GitHub Repo: <https://github.com/maxslagt/assignment8>


### Data  

The training data for this project are available at: 

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available at: 

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

### Goal

The goal of this project is to predict the manner in which subjects did 
the exercise.  

### Load the functions and static variables
All functions are loaded and static variables are assigned.  Also in this 
section, the seed is set so the pseudo-random number generator operates in a 
consistent way for repeat-ability.  


```r
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(e1071)
library(randomForest)
set.seed(1)
train.url <-
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test.url <- 
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
path <- paste(getwd(),"/", "machine", sep="")
train.file <- file.path(path, "machine-train-data.csv")
test.file <- file.path(path, "machine-test-data.csv")
```

### Dowload the files (if necessary) and read them into memory  
The files are read into memory.  Various indicators of missing data (i.e., 
"NA", "#DIV/0!" and "") are all set to NA so they can be processed.  


```r
if (!file.exists(train.file)) {
        download.file(train.url, destfile=train.file)
}
if (!file.exists(test.file)) {
        download.file(test.url, destfile=test.file)
}
train.data.raw <- read.csv(train.file, na.strings=c("NA","#DIV/0!",""))
test.data.raw <- read.csv(test.file, na.strings=c("NA","#DIV/0!",""))
```

### Remove unecessary colums
Columns that are not deeded for the model and columns that contain NAs 
are eliminated.  


```r
# Drop the first 7 columns as they're unnecessary for predicting.
train.data.clean1 <- train.data.raw[,8:length(colnames(train.data.raw))]
test.data.clean1 <- test.data.raw[,8:length(colnames(test.data.raw))]
# Drop colums with NAs
train.data.clean1 <- train.data.clean1[, colSums(is.na(train.data.clean1)) == 0] 
test.data.clean1 <- test.data.clean1[, colSums(is.na(test.data.clean1)) == 0] 
# Check for near zero variance predictors and drop them if necessary
nzv <- nearZeroVar(train.data.clean1,saveMetrics=TRUE)
zero.var.ind <- sum(nzv$nzv)
if ((zero.var.ind>0)) {
        train.data.clean1 <- train.data.clean1[,nzv$nzv==FALSE]
}
```

### Slice the data for cross validation  
The training data is divided into two sets.  This first is a training set with 70% of the data which is used to train the model.  The second is a validation 
set used to assess model performance.  


```r
in.training <- createDataPartition(train.data.clean1$classe, p=0.70, list=F)
train.data.final <- train.data.clean1[in.training, ]
validate.data.final <- train.data.clean1[-in.training, ]
```

# Model Development  
### Train the model  
The training data-set is used to fit a Random Forest model because it 
automatically selects important variables and is robust to correlated 
covariates & outliers in general. 5-fold cross validation is used when 
applying the algorithm. A Random Forest algorithm is a way of averaging 
multiple deep decision trees, trained on different parts of the same data-set,
with the goal of reducing the variance. This typically produces better 
performance at the expense of bias and interpret-ability. The Cross-validation 
technique assesses how the results of a statistical analysis will generalize 
to an independent data set. In 5-fold cross-validation, the original sample 
is randomly partitioned into 5 equal sized sub-samples. a single sample 
is retained for validation and the other sub-samples are used as training 
data. The process is repeated 5 times and the results from the folds are 
averaged.


```r
control.parms <- trainControl(method="cv", 5)
rf.model <- train(classe ~ ., data=train.data.final, method="rf",
                 trControl=control.parms, ntree=251)
rf.model
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10990, 10990, 10989, 10990, 10989 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9907547  0.9883044
##   27    0.9909733  0.9885816
##   52    0.9820921  0.9773451
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

### Estimate performance  
The model fit using the training data is tested against the validation data.
Predicted values for the validation data are then compared to the actual 
values. This allows forecasting the accuracy and overall out-of-sample error,
which indicate how well the model will perform with other data.  


```r
rf.predict <- predict(rf.model, validate.data.final)
confusionMatrix(validate.data.final$classe, rf.predict)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1669    2    2    0    1
##          B    6 1130    2    1    0
##          C    0    4 1019    3    0
##          D    0    0    4  958    2
##          E    0    2    1    3 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9944          
##                  95% CI : (0.9921, 0.9961)
##     No Information Rate : 0.2846          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9929          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9930   0.9912   0.9927   0.9972
## Specificity            0.9988   0.9981   0.9986   0.9988   0.9988
## Pos Pred Value         0.9970   0.9921   0.9932   0.9938   0.9945
## Neg Pred Value         0.9986   0.9983   0.9981   0.9986   0.9994
## Prevalence             0.2846   0.1934   0.1747   0.1640   0.1833
## Detection Rate         0.2836   0.1920   0.1732   0.1628   0.1828
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9976   0.9955   0.9949   0.9958   0.9980
```

```r
accuracy <- postResample(rf.predict, validate.data.final$classe)
acc.out <- accuracy[1]
overall.ose <- 
        1 - as.numeric(confusionMatrix(validate.data.final$classe, rf.predict)
                       $overall[1])
```

### Results  
The accuracy of this model is **0.9943925** and the Overall Out-of-Sample 
error is **0.0056075**.

# Run the model
The model is applied to the test data to produce the results.


```r
results <- predict(rf.model, 
                   test.data.clean1[, -length(names(test.data.clean1))])
results
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

# Appendix - Decision Tree Visualization


```r
treeModel <- rpart(classe ~ ., data=train.data.final, method="class")
fancyRpartPlot(treeModel)
```

![](assignment8_files/figure-html/unnamed-chunk-8-1.png)<!-- -->
