#Installing Packages
install.packages("e1071")
install.packages("caTools")
install.packages("caret")
install.packages('ROCR')
install.packages("smotefamily")
install.packages('ROSE')

# Loading package
library("e1071")
library("caTools")
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
library("naivebayes")
library("mlbench")

################################Load data#######################################
dataCSV <- read.csv("./data/dev.csv")
dataCSV <- na.omit(dataCSV)

##################### Analyze our data set and fix unbalanced ##################
# Re-sample the data in 55000 samples
# The following variables have a variance of 0
tem.dataCSV.balanced.sample <- subset(dataCSV, select=-c(ID,
                                                         Parent,
                                                         Component,
                                                         Line,
                                                         Column,
                                                         EndLine,
                                                         EndColumn,
                                                         WarningBlocker, 
                                                         Code.Size.Rules, 
                                                         Comment.Rules, 
                                                         Coupling.Rules, 
                                                         MigratingToJUnit4.Rules,
                                                         Migration13.Rules,
                                                         Migration14.Rules,
                                                         Migration15.Rules,
                                                         Vulnerability.Rules))
tem.dataCSV.balanced.sample <- na.omit(tem.dataCSV.balanced.sample)
tem.dataCSV.balanced.sample <- tem.dataCSV.balanced.sample[1:55000, ]
tem.dataCSV.balanced.sample$MigratingToJUnit4.Rules # should be NULL
# Turn bugs logical column into a integer value of 1 or 0.
tem.dataCSV.balanced.sample$bugs <-
  as.integer(as.logical(tem.dataCSV.balanced.sample$bugs))

# Visualize the data
barplot(prop.table(table(tem.dataCSV.balanced.sample$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Bugs class distribution")

################################################################################
set.seed(2987465)
# createDataPartition() function from the caret package to split the original 
# data set into a training and testing set and split data into training 
# (70%) and testing set (30%)
parts = createDataPartition(tem.dataCSV.balanced.sample$bugs, 
                            p = 0.70, 
                            list = FALSE)

tem.dataCSV.train = tem.dataCSV.balanced.sample[parts, ]
tem.dataCSV.test = tem.dataCSV.balanced.sample[-parts, ]

tem.dataCSV.train = tem.dataCSV.train[complete.cases(tem.dataCSV.train), ]
tem.dataCSV.test = tem.dataCSV.test[complete.cases(tem.dataCSV.test), ]

X_train = tem.dataCSV.train[,-ncol(tem.dataCSV.balanced.sample)]
y_train = tem.dataCSV.train[,ncol(tem.dataCSV.balanced.sample)]

dim(tem.dataCSV.train)
dim(tem.dataCSV.test)

################################################################################
# The SMOTE function requires the target variable to be numeric
tem.dataCSV.train$bugs <- as.numeric(tem.dataCSV.train$bugs)
tem.dataCSV.test$bugs <- as.numeric(tem.dataCSV.test$bugs)
class(tem.dataCSV.train$bugs)
class(tem.dataCSV.test$bugs)
# It is a numeric now!

# For the training data set
# All but the last column
tem.dataCSV.train.SMOTE <- SMOTE(tem.dataCSV.train[,-ncol(tem.dataCSV.train)],
                                 tem.dataCSV.train$bugs,
                                 K = 5)

# Extract only the balanced data set
tem.dataCSV.train.SMOTE <- tem.dataCSV.train.SMOTE$data
# Change the name from class to bugs
colnames(tem.dataCSV.train.SMOTE) [ncol(tem.dataCSV.train.SMOTE)] <- "bugs"
tem.dataCSV.train.SMOTE$bugs <- as.factor(tem.dataCSV.train.SMOTE$bugs)
table(tem.dataCSV.train.SMOTE$bugs)

train.smote.both <- 
  ovun.sample(bugs~., data=tem.dataCSV.train.SMOTE, 
              method = "both")$data
table(train.smote.both$bugs)

# Visualize the data
barplot(prop.table(table(train.smote.both$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Bugs class distribution (SMOTE, Over, Under)")

################################################################################
############################# Feature selection ################################
# Define the control using a naive Bayes selection function
# Change for other functions for classification problems, see the link bellow:
# http://topepo.github.io/caret/recursive-feature-elimination.html#rfe
# Run the RFE algorithm
################################################################################
# Turn last column of the data set into a factor
# To run the naive Bayes algorithm

class(tem.dataCSV.train[,ncol(tem.dataCSV.balanced.sample)])

tem.dataCSV.train[,ncol(tem.dataCSV.balanced.sample)] <- 
  as.factor(tem.dataCSV.train[,ncol(tem.dataCSV.balanced.sample)]) 

# SHOULD BE A FACTOR
class(tem.dataCSV.train[,ncol(tem.dataCSV.balanced.sample)])
################################################################################
control <- rfeControl(functions = nbFuncs,
                      method = "repeatedcv",
                      number = 10,
                      verbose = TRUE)

subsets <- c(1:ncol(train.smote.both), 10, 15, 20, 25)
################################################################################
# Run the RFE algorithm
################################################################################
results <- rfe(x = X_train, 
               y = y_train, 
               sizes = subsets, 
               rfeControl = control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))
variables <- data.frame(results$optVariables) # STORE THE VARIABLES
variables$results.optVariables

##################### Apply the naive Bayes algorithm ##########################
############################## Variable analysis ###############################
# Pearson test of 0.214769 not very high
#
#
# The Response for Class (RFC) metric is the total number of methods that 
# can potentially be executed in response to a message 
# received by an object of a class. 
#
# This number is the sum of the methods of the class, 
# and all distinct methods are invoked directly within the class methods. 
# Additionally, inherited methods are counted, 
# but overridden methods are not, 
# because only one method of a particular signature 
# will always be available to an object of a given class.
# 
# Despite the lower correlation, 
# it makes sense that this metric makes affects the Naïve-Bayes algorithm
################################################################################
cor.test(tem.dataCSV.balanced.sample$bugs, 
         tem.dataCSV.balanced.sample$RFC, 
         method = "pearson")


model <- naive_bayes(bugs ~ RFC, 
                     data = train.smote.both,
                     usekernel = T) 
plot(model)

model$tables

predicted <- predict(model, 
                     train.smote.both, 
                     type = 'prob')
head(cbind(predicted,
           train.smote.both))


predicted.1 <- predict(model, train.smote.both)
# Confusion Matrix

cm <- table(predicted.1,
            train.smote.both$bugs)
cm

# Model Evaluation
confusionMatrix(cm)

misclassification_Traning <- 1-sum(diag(cm))/sum(cm)
misclassification_Traning
# Misclassification is around 22%%.
# Training model accuracy is around 78% !


############################ Test data #########################################
p2 <- predict(model, tem.dataCSV.test, type = 'prob')

cm2 <- table(p2, tem.dataCSV.test$bugs)
cm2
# Model Evaluation
confusionMatrix(cm2)

misclassification_Testing <- 1 - sum(diag(cm2))/sum(cm2)
misclassification_Testing

## Misclassification is around 15%%.

## Testing model accuracy is around 85% not bad!.

# ROC - are under the curve a more viable metric for the accuracy of our model
# Both of the arguments on the ROC function need to be numeric!
class(tem.dataCSV.test$bugs)
dt.preds <- as.numeric(p2)
roc.accuracy <- roc(tem.dataCSV.test$bugs, dt.preds)
print(roc.accuracy)
plot(roc.accuracy)
# The area under the curve is 0.7044

################################################################################
dataCSVComp <- read.csv("data/comp.csv")
dataCSVComp <- na.omit(dataCSVComp)
dataCSVComp <- subset(dataCSVComp, select=-c(ID, Parent, Component, Line,
                                             Column, EndLine, EndColumn,
                                             WarningBlocker, Code.Size.Rules,
                                             Comment.Rules, Coupling.Rules,
                                             MigratingToJUnit4.Rules,
                                             Migration13.Rules,
                                             Migration14.Rules,
                                             Migration15.Rules,
                                             Vulnerability.Rules))

####################### Naive Bayes with competition data ######################
# Prediction
predict.bug <- predict(model, data = dataCSVComp, type = 'prob')
predict.bug <- predict.bug[-c(1)]

# Save the data into submission format.
write.csv(predict.bug, "submissions/naive_bayes.csv")
