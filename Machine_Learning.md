Summary
=======

Using tracking devices such as Jawbone Up, Nike FuelBand, and Fitbit people who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants will be used to classify how well the exercise has been performed. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. Details on the data can be found here: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har> (Weight Lifting Exercise Dataset).

The training data set used was: <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data set used was:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The test data provided was only used after cross validating the model on the training data. This was done by splitting the training data into new training and testing data sets

The model has a 98.5% accuracy on the new training set, 99% accuarcy on the new testing data set and 100% corrrect for the Quiz data set(quiz answers checked externally).

Modelling
---------

### Downloading the Data

``` r
download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv','training.csv')
traindata<-read.csv('training.csv', na.strings=c("NA","#DIV/0!",""))
traindata<-as.data.frame(traindata)
```

``` r
download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv','test.csv')
testdata<-read.csv('test.csv', na.strings=c("NA","#DIV/0!",""))
testdata<-as.data.frame(testdata)
```

### Data Exploration and Preprocessing

The data has 160 variables which is a large amount to try use in a model. Therefore after looking at the data, 3 things were noticed: 1. The first 7 columns are variables that will not affect the outcome 2. There are a lot of variables that do no vary significantly 3. There are a lot of variables which mostly have NA values

To deal with these 3 features of the data, the data was processed in 3 ways: 1. The first 7 columns were removed 2. Near Zero Variance analysis was used to remove 101 variables 3. any columns still with NA values were removed.

The training data was then split into a smaller training data set and a new testing set to cross validate the model before trying it on the quiz data provided.

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
set.seed(33221)
dim((traindata))
```

    ## [1] 19622   160

``` r
##Remove top first 7, which are not valid variables.
c<-c(1:7)
traindata1<-traindata[,-c]
testdata1<-testdata[,-c]
##checking for near zero variability in the variables
nsv<-nearZeroVar(testdata,freqCut=2,uniqueCut=20,saveMetrics = F)
traindata2<-traindata1[,-nsv]
testdata2<-testdata1[,-nsv]
##checking for columns with NA values and removing those columns from the variables
traindata3<-traindata2[,names(traindata2[,colSums(is.na(traindata2)) == 0])]
names(testdata2)[names(testdata2) == "problem_id"] <- "classe"
testdata3<-testdata2[,names(traindata2[,colSums(is.na(traindata2)) == 0])]
##Partitioning the training set to do cross validation
inTrain <- createDataPartition(traindata3$classe, p=0.6, list=FALSE)
training <- traindata3[inTrain,]
testing <- traindata3[-inTrain,]
dim(training); dim(testing);
```

    ## [1] 11776    20

    ## [1] 7846   20

Above are the dimensions of the newly created training and testing set. \#\#\#Creating the Model The first model fitted oto the data was Random Forest. This was tried first as it was the method used in by Velloso et al., 2013 in the original study.

``` r
library(parallel)
library(doParallel)
```

    ## Loading required package: foreach

    ## Loading required package: iterators

``` r
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
fitControl <- trainControl(method = "cv",number = 5, allowParallel = TRUE)    
model<-train(classe~.,method="rf",data=training,trcontrol=fitControl)
```

``` r
model
```

    ## Random Forest 
    ## 
    ## 11776 samples
    ##    19 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9854326  0.9815788
    ##   10    0.9847222  0.9806817
    ##   19    0.9783679  0.9726485
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 2.

### Cross Validating the Model

``` r
predictionCV<-predict(model,testing)
confusionMatrix(predictionCV,testing$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2229   10    1    0    0
    ##          B    2 1503   14    0    2
    ##          C    0    4 1350   17    1
    ##          D    0    1    3 1269    5
    ##          E    1    0    0    0 1434
    ## 
    ## Overall Statistics
    ##                                        
    ##                Accuracy : 0.9922       
    ##                  95% CI : (0.99, 0.994)
    ##     No Information Rate : 0.2845       
    ##     P-Value [Acc > NIR] : < 2.2e-16    
    ##                                        
    ##                   Kappa : 0.9902       
    ##  Mcnemar's Test P-Value : NA           
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9987   0.9901   0.9868   0.9868   0.9945
    ## Specificity            0.9980   0.9972   0.9966   0.9986   0.9998
    ## Pos Pred Value         0.9951   0.9882   0.9840   0.9930   0.9993
    ## Neg Pred Value         0.9995   0.9976   0.9972   0.9974   0.9988
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2841   0.1916   0.1721   0.1617   0.1828
    ## Detection Prevalence   0.2855   0.1939   0.1749   0.1629   0.1829
    ## Balanced Accuracy      0.9983   0.9936   0.9917   0.9927   0.9971

### Predicting Results for the Quiz Data

``` r
prediction<-predict(model,testdata3)
```

References
----------

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>
