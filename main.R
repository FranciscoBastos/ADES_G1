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
# We are going to try with over, under an both fittings as well as with 
# ROSE and SMOTE to see what gives the best accuracy.
# 
###########################Try with over fitting################################
# We need to do over sampling for all the samples!!!
# We need to do under sampling for all the samples!!!
# We need to do both (over and under) sampling for all the samples!!!
# We need to do SMOTE sampling for all the samples!!!
################################### Over #######################################

balanced.data.over.sampling <- 
        ovun.sample(bugs~., data=tem.dataCSV.balanced.sample, 
                    method = "over")$data
table(balanced.data.over.sampling$bugs)

# Visualize the data
barplot(prop.table(table(balanced.data.over.sampling$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution over sampling")

#################################### Under #####################################

balanced.data.under.sampling <- 
        ovun.sample(bugs~., data=tem.dataCSV.balanced.sample, 
                    method = "under")$data
table(balanced.data.under.sampling$bugs)

# Visualize the data
barplot(prop.table(table(balanced.data.under.sampling$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution under sampling")

################################### Both #######################################

balanced.data.both.sampling <- 
        ovun.sample(bugs~., data=tem.dataCSV.balanced.sample, 
                    method = "both")$data
table(balanced.data.both.sampling$bugs)

# Visualize the data
barplot(prop.table(table(balanced.data.both.sampling$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution over and under sampling")

############################### SMOTE function #################################
# TODO SMOTE FUNCTION NOT WORKING YET
################################################################################
balanced.data.SMOTE.sampling <- 
        SMOTE(tem.dataCSV.balanced.sample$bugs, tem.dataCSV.balanced.sample$bugs)
table(balanced.data.SMOTE.sampling$bugs)

# Visualize the data
barplot(prop.table(table(balanced.data.SMOTE.sampling$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution ROSE function")

###########################Divide our balanced data ############################
# Trying with: both
#
################################################################################

set.seed(2987465)
index <- sample(1:nrow(balanced.data.both.sampling), 
                as.integer(0.7*nrow(balanced.data.both.sampling)))
tem.dataCSV.train <- balanced.data.both.sampling[index,]
tem.dataCSV.test <- balanced.data.both.sampling[-index,]

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

# Compute the confusion matrix and other important data!
# TODO NOT WORKING BECAUSE THE DATA IS AS AN INTEGER INSTED AS AN FACTOR
confusionMatrix(predict(dt, tem.dataCSV.test, type = "class"), 
                tem.dataCSV.test$bugs)
# Very high accuracy: 
#                       Accuracy : 0.9901          
#                       95% CI : (0.9887, 0.9913)
#                       No Information Rate : 0.9534          
#                       P-Value [Acc > NIR] : < 2.2e-16
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
predict.bug <- predict(dt, tem.dataCSVComp.balanced, type = 'prob')
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

