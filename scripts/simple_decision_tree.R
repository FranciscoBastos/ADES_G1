install.packages('abind')
install.packages('zoo')
install.packages('xts')
install.packages('quantmod')
install.packages('ROCR')
install.packages("DMwR")
install.packages("smotefamily")
install.packages('caret')
install.packages('ROSE')
install.packages("gdata", "tree", "ROCR")

library("caret")
library("ROCR")
library("psych")
library("rpart")
library("dplyr")
library("ggpubr")
library("corrplot")
library("RColorBrewer")
library("readxl")
library("ggplot2")
library("rpart.plot")
library("neuralnet")
library("party")
library("partykit")
library("DMwR")
library("DMwR2")
library("smotefamily")
library("ROSE")
library("tree")
library("gdata")
library("ROCR")
library("tidyverse")
library("leaps")
library("pROC")

suppressPackageStartupMessages(c(library(caret),library(corrplot),library(smotefamily)))

if(!require('DMwR')) {
  install.packages('DMwR')
  library('DMwR')
}

####################################HELPS#######################################
# https://github.com/sed-inf-u-szeged/OpenStaticAnalyzer

################################Load data#######################################

dataCSV <- read.csv("./data/dev.csv")

##############################Primary analysis##################################

summary(dataCSV)
head(dataCSV)
tail(dataCSV)
dim(dataCSV)

dataCSV <- na.omit(dataCSV)

############################K-Means clustering algorithm########################
# For more information, if there is an error go-to:
# https://data-hacks.com/r-error-do_one-nmeth-na-nan-inf-foreign-function-call-arg-1
################################################################################
# Create a vector for analysis with only the first 25000 lines of the data-set.
# Also we removed the 1 to the 7 columns - because they were labeling columns.
#
################################################################################
tem.dataCSV <- dataCSV[1:25000,-c(1:7)]

# Made this to turn the logical values of true and false to 1 and 0 respectively
tem.dataCSV$bugs <- as.integer(as.logical(tem.dataCSV$bugs))

# K-Means algorithm from the simplified data-set.
kmeans.tem.dataCSV <- kmeans(tem.dataCSV, centers=2)
print(kmeans.tem.dataCSV)

#################################Divide our data################################

set.seed(2987465)
index <- sample(1:nrow(tem.dataCSV), as.integer(0.7*nrow(tem.dataCSV)))
tem.dataCSV.train <- tem.dataCSV[index,]
tem.dataCSV.test <- tem.dataCSV[-index,]

dim(tem.dataCSV.train)
dim(tem.dataCSV.test)

tem.dataCSV.train <- na.omit(tem.dataCSV.train)
tem.dataCSV.test <- na.omit(tem.dataCSV.test)

#################################Decision tree##################################
dt <- rpart( bugs ~ .,
             data = tem.dataCSV.train,
             method = "class")
# We use the type = "class" for the classification tree
dt.preds <- predict(dt, tem.dataCSV.train, type = "class")
# We use the type = "prob" to get the probabilities
dt.preds
dt.pred.probs <- predict(dt, tem.dataCSV.train, type = "prob") 
dt.pred.probs

# make predictions for test data
dt.preds <- predict(dt, tem.dataCSV.test, type="class")

# compute confusion matrix
cm.dt <- table(dt.preds, tem.dataCSV.test$bugs)
cm.dt

accuracy <- sum(diag(cm.dt)) / sum(cm.dt)
accuracy
error.dt.test <- 1 - sum(diag(cm.dt)) / sum(cm.dt)
error.dt.test

# ROC - are under the curve a more viable metric for the accuracy of our model
# Both of the arguments on the ROC function need to be numeric!
library(pROC)
class(tem.dataCSV.test$bugs)
dt.preds <- as.numeric(dt.preds)
class(dt.preds)
roc.accuracy <- roc(tem.dataCSV.test$bugs, dt.preds)
print(roc.accuracy)
plot(roc.accuracy)