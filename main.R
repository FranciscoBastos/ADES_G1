install.packages('abind')
install.packages('zoo')
install.packages('xts')
install.packages('quantmod')
install.packages('ROCR')
install.packages("DMwR")
install.packages("smotefamily")
install.packages('caret')

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

suppressPackageStartupMessages(c(library(caret),library(corrplot),library(smotefamily)))

####################################HELPS#######################################
# https://github.com/sed-inf-u-szeged/OpenStaticAnalyzer

################################Load data#######################################

dataCSV <- read.csv("data/dev.csv")

##############################Primary analysis##################################

summary(dataCSV)
head(dataCSV)
tail(dataCSV)
str(dataCSV)
dim(dataCSV)

dataCSV <- na.omit(dataCSV)

############################K-Means clustering algorithm########################
# For more information, if there is an error go-to:
# https://data-hacks.com/r-error-do_one-nmeth-na-nan-inf-foreign-function-call-arg-1
################################################################################
# Create a vector for analysis with only the first 1000 lines of the data-set.
# Also we removed the 1 to the 3 columns and the last - because they had 
# non numerical values.
#
# Made this to turn the logical values of true and false to 1 and 0 respectively
# Remove the last column as well: 
# tem.dataCSV <- dataCSV[1:1000,-c(1:7,ncol(dataCSV))]
tem.dataCSV <- dataCSV[1:25000,-c(1:7)]

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

dt.preds <- predict(dt, tem.dataCSV.train, type = "class") # class
dt.pred.probs <- predict(dt, tem.dataCSV.train, type = "prob") # ... and probabilities
dt.pred.probs

# make predictions for test data
dt.preds <- predict(dt, tem.dataCSV.test, type="class") # class

# compute confusion matrix
cm.dt <- table(dt.preds, tem.dataCSV.test$bugs)
cm.dt

accuracy <- sum(diag(cm.dt)) / sum(cm.dt)
accuracy
error.dt.test <- 1 - sum(diag(cm.dt)) / sum(cm.dt)
error.dt.test

###################################### ROC #####################################
# TODO ROC NOT WORKING - FIX LATER, ERROR MAYBE ON THE PREDICTION ARGUMENTS FEED
roc.stats <- prediction(dt.preds[, "1"], tem.dataCSV.train$bugs)
roc <- performance(roc.stats, "tpr", "fpr")

################################################################################

# ATENTION: YOU NEED TO USE THE COMPETION DATA, THE ALL DATASET.
#           YOU CANNOT DEVIDE THE DATASET INTO TRAINING AND TESTING.
#           IT WILL NOT WORK, TRUST ME!
# 
# ALSO, LEARN HOW TO USE THE COLLECT FUNCTION 
# TO SELECT A CORRECT BETTER IMPUT DATA FRAME
# NOTE: THE ACCURACY IS 0.618431 WITH THE CLASSIFICATION TREE!
#       NEXT TIME TRY TO IMPROVE.
#
# Cannot calculate confusion matrix, error and accuracy
# because the bugs column does not exist.

# Export data set to more or less Kaggle format

dataCSVComp <- read.csv("data/comp.csv")
dataCSVComp <- na.omit(dataCSVComp)
tem.dataCSVComp <- dataCSVComp[-c(1:7)]

#######################Decision tree with competition data######################

# Prediction
predict.bug <- predict(dt, tem.dataCSVComp, type = 'prob')
predict.bug

# make predictions for training data
dt.preds <- predict(dt, tem.dataCSVComp, type="class") # class

# Turn prediction data into a data.frame
df.predict.bug <- data.frame(predict.bug)
# Eliminate the first column because it is the classes that do not have bugs
df.predict.bug <- df.predict.bug[,-c(1)]
df.predict.bug

predict.bug <- predict.bug[,-c(-2)]
predict.bug

# Save the data into submission format.

write.csv(predict.bug, "submissions/decision_tree_formated.csv",
          row.names = TRUE,
          col.names = TRUE)

###############################Cleaning the data ###############################
# How to correct the data test information, when we know that the data is 
# unbalanced?
# 
# - We can re-dimension the data sample.
# - Or we can create symmetric data using SMOTE or GANs algorithms.
#
#
# The team is going to try to sample the test data
# to get a more trustworthy prediction.
#
# Helps:
# https://mikedenly.com/posts/2020/03/balanced-panel/
# https://www.r-bloggers.com/2021/05/class-imbalance-handling-imbalanced-data-in-r/ 
# 
################################################################################
##################### Analyze our data set and fix unbalanced ##################

if(!require('ROSE')) {
        install.packages('ROSE')
        library('ROSE')
}

suppressMessages(library(plm)) # to remove note about dependencies
is.pbalanced(tem.dataCSV)
# Is the data set balanced?
# $ FALSE
# The data set is not balanced!

suppressMessages(library(tidyverse))

tem.dataCSV.balanced <- make.pbalanced(tem.dataCSV,
                                       balance.type = c("fill"))

is.pbalanced(tem.dataCSV.balanced)
# Is the data set balanced?
# $ TRUE
# The data set is balanced!

tem.dataCSV.balanced <- na.omit(dataCSV)

table(tem.dataCSV.balanced$bugs)
# Are the bugs balanced ?
# No the bugs are not balanced.
# Let's balance our bugs.

tem.dataCSV.balanced$bugs <- as.factor(as.logical(tem.dataCSV.balanced$bugs))

# Visualize the data
barplot(prop.table(table(tem.dataCSV.balanced$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution")

###########################Divide our balanced data ############################

set.seed(2987465)
index <- sample(1:nrow(tem.dataCSV.balanced), 
                as.integer(0.7*nrow(tem.dataCSV.balanced)))
tem.dataCSV.train <- tem.dataCSV.balanced[index,]
tem.dataCSV.test <- tem.dataCSV.balanced[-index,]

dim(tem.dataCSV.train)
dim(tem.dataCSV.test)

under <- ovun.sample(bugs~., data=tem.dataCSV.train, method = "under")$data
table(under$bugs)

# Visualize the data
barplot(prop.table(table(under$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution")

############################ Prediction with the under data ####################

dt <- rpart( bugs ~ .,
             data = under,
             method = "class")

dt.preds <- predict(dt, under, type = "class") # class
dt.pred.probs <- predict(dt, under, type = "prob") # ... and probabilities
dt.pred.probs

# compute confusion matrix
cm.dt <- table(dt.preds, under$bugs)
cm.dt

accuracy <- sum(diag(cm.dt)) / sum(cm.dt)
accuracy
error.dt.test <- 1 - sum(diag(cm.dt)) / sum(cm.dt)
error.dt.test

