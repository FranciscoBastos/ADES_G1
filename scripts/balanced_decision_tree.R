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
# https://rpubs.com/ZardoZ/SMOTE_FRAUD_DETECTION
#
################################################################################
##################### Analyze our data set and fix unbalanced ##################

suppressMessages(library(plm)) # to remove note about dependencies
is.pbalanced(tem.dataCSV)
# Is the data set balanced?
# $ FALSE
# The data set is not balanced!

suppressMessages(library(tidyverse))

tem.dataCSV.balanced <- make.pbalanced(tem.dataCSV,
                                       balance.type = c("fill"))

class(tem.dataCSV.balanced$bugs)

# Turn bugs logical column into a integer value of 1 or 0.
tem.dataCSV.balanced$bugs <-
  as.integer(as.logical(tem.dataCSV.balanced$bugs))

is.pbalanced(tem.dataCSV.balanced)
# Is the data set balanced?
# $ TRUE
# The data set is balanced!

# Re-sample the data in half
tem.dataCSV.balanced.sample <- tem.dataCSV.balanced[1:35000, ]
tem.dataCSV.balanced.sample <- na.omit(tem.dataCSV.balanced.sample)
# Turn bugs logical column into a integer value of 1 or 0.
tem.dataCSV.balanced.sample$bugs <-
  as.integer(as.logical(tem.dataCSV.balanced.sample$bugs))

table(tem.dataCSV.balanced.sample$bugs)
# Are the bugs balanced ?
# No the bugs are not balanced.
# Let's balance our bugs.

# Visualize the data
barplot(prop.table(table(tem.dataCSV.balanced.sample$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution")

################################################################################
# We are going to try with over, under an both sampling as well as with 
# SMOTE function to see what gives the best accuracy.
# 
############################## Try with sampling ###############################
# We need to do over sampling for all the samples!!!
# We need to do under sampling for all the samples!!!
# We need to do both (over and under) sampling for all the samples!!!
################################### Over #######################################
set.seed(2987465)
index <- sample(1:nrow(tem.dataCSV.balanced.sample), 
                as.integer(0.7*nrow(tem.dataCSV.balanced.sample)))
tem.dataCSV.train <- tem.dataCSV.balanced.sample[index,]
tem.dataCSV.test <- tem.dataCSV.balanced.sample[-index,]

dim(tem.dataCSV.train)
dim(tem.dataCSV.test)


balanced.data.over.sampling <- 
  ovun.sample(bugs~., data=tem.dataCSV.train, 
              method = "over")$data
table(balanced.data.over.sampling$bugs)

# Visualize the data
barplot(prop.table(table(balanced.data.over.sampling$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution over sampling")

#################################### Under #####################################
set.seed(2987465)
index <- sample(1:nrow(tem.dataCSV.balanced.sample), 
                as.integer(0.7*nrow(tem.dataCSV.balanced.sample)))
tem.dataCSV.train <- tem.dataCSV.balanced.sample[index,]
tem.dataCSV.test <- tem.dataCSV.balanced.sample[-index,]

dim(tem.dataCSV.train)
dim(tem.dataCSV.test)

balanced.data.under.sampling <- 
  ovun.sample(bugs~., data=tem.dataCSV.train, 
              method = "under")$data
table(balanced.data.under.sampling$bugs)

# Visualize the data
barplot(prop.table(table(balanced.data.under.sampling$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution under sampling")

################################### Both #######################################
set.seed(2987465)
index <- sample(1:nrow(tem.dataCSV.balanced.sample), 
                as.integer(0.7*nrow(tem.dataCSV.balanced.sample)))
tem.dataCSV.train <- tem.dataCSV.balanced.sample[index,]
tem.dataCSV.test <- tem.dataCSV.balanced.sample[-index,]

dim(tem.dataCSV.train)
dim(tem.dataCSV.test)

balanced.data.both.sampling <- 
  ovun.sample(bugs~., data=tem.dataCSV.train, 
              method = "both")$data
table(balanced.data.both.sampling$bugs)

# Visualize the data
barplot(prop.table(table(balanced.data.both.sampling$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution over and under sampling")
###########################Divide our balanced data ############################
# Trying with: both
#
################################################################################

dim(tem.dataCSV.train)
dim(tem.dataCSV.test)

##################### Prediction with the balanced data ########################

dt <- rpart( bugs ~ .,
             data = tem.dataCSV.train,
             method = "class")

# Use the type = "class" for a classification tree!
dt.preds <- predict(dt, tem.dataCSV.test, type = "class")
# Use the type = "prob" to get the probabilities!
dt.pred.probs <- predict(dt, tem.dataCSV.test, type = "prob")
dt.pred.probs

# compute confusion matrix
cm.dt <- table(dt.preds, tem.dataCSV.test$bugs)
cm.dt

accuracy <- sum(diag(cm.dt)) / sum(cm.dt)
accuracy
error.dt.test <- 1 - sum(diag(cm.dt)) / sum(cm.dt)
error.dt.test

class(tem.dataCSV.test$bugs)
dt.preds <- as.numeric(dt.preds)
class(dt.preds)
roc.accuracy <- roc(tem.dataCSV.test$bugs, dt.preds)
print(roc.accuracy)
plot(roc.accuracy)
# The are under the curve improved a lot!
# Area under the curve:  0.5377