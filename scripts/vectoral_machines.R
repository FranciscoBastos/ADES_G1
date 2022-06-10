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
install.packages("gdata")
install.packages("RandomForestsGLS", dependencies = TRUE)
install.packages("e1071")
install.packages("ggstatsplot")

library("e1071")
library("gdata")
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
library("DMwR2")
library("smotefamily")
library("ROSE")
library("tree")
library("gdata")
library("ROCR")
library("tidyverse")
library("leaps")
library("pROC")
library("ggplot2")
library("randomForestExplainer")
library("randomForestSRC")
library("keras")
library("mlbench")
library("magrittr")
library("e1071")
library("rgl")
library("misc3d")
library("ggstatsplot")
library("outliers")
library("doMC")
library("ElemStatLearn")


install_tensorflow()
use_condaenv("r-tensorflow")

require(caTools)

suppressPackageStartupMessages(c(library(caret),
                                 library(corrplot),
                                 library(smotefamily)))

################################Load data#######################################

dataCSV <- read.csv("./data/dev.csv")
dataCSV <- na.omit(dataCSV)
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
##################### Analyze our data set and fix unbalanced ##################

# Visualize the data
barplot(prop.table(table(tem.dataCSV.balanced.sample$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class distribution")
############################### Remove the outliers ############################
# https://rdrr.io/cran/outliers/man/rm.outlier.html
################################################################################
boxplot(tem.dataCSV.balanced.sample)$out

rm.outlier(tem.dataCSV.balanced.sample, fill = TRUE)

tem.dataCSV.balanced.sample

################################################################################
############################### SMOTE function #################################
# Divide our balanced data 
# Trying with: SMOTE
#
################################################################################
set.seed(2987465)
index <- sample(1:nrow(tem.dataCSV.balanced.sample), 
                as.integer(0.7*nrow(tem.dataCSV.balanced.sample)))
tem.dataCSV.train <- tem.dataCSV.balanced.sample[index,]

tem.dataCSV.test <- tem.dataCSV.balanced.sample[-index,]

dim(tem.dataCSV.train)
dim(tem.dataCSV.test)

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
################################### Both #######################################
# To decrease the size of the data-set we should use under
# Needed this to reduce the size of the data set
# https://www.statology.org/smote-in-r/
################################################################################
tem.dataCSV.train.SMOTE <- tem.dataCSV.train.SMOTE[1:55000, ]
################################################################################

train.smote.both <- 
  ovun.sample(bugs~., data=tem.dataCSV.train.SMOTE, 
              method = "both")$data
table(train.smote.both$bugs)

# Visualize the data
barplot(prop.table(table(train.smote.both$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class distribution (SMOTE, Over, Under)")

############################ Normalize the features ############################
# https://medium.com/0xcode/svm-classification-algorithms-in-r-ced0ee73821
#
train.smote.both[-ncol(train.smote.both)] = 
  scale(train.smote.both[-ncol(train.smote.both)])

tem.dataCSV.test[-ncol(tem.dataCSV.test)] = 
  scale(tem.dataCSV.test[-ncol(tem.dataCSV.test)])

##################################### SVM ######################################

svm.model = svm(formula = bugs ~ .,
                data = train.smote.both,
                type = 'C-classification',
                kernel = 'polynomial') # change to linear later

plot(svm.model, train.smote.both, bugs ~ RFC)

############################# Predict and print the SVM ########################

svm.predict <- predict(svm.model, newdata = tem.dataCSV.test)

svm.predict

cm <- table(tem.dataCSV.test[, ncol(tem.dataCSV.test)], svm.predict)
cm
accuracy <- sum(diag(cm)) / sum(cm)
accuracy # 0.5595152
error.dt.test <- 1 - sum(diag(cm)) / sum(cm)
error.dt.test # 0.4404848

precision <- cm[1, 1]/sum(cm[,1])
precision # 0.9901013
recall <- cm[1, 1]/sum(cm[1,])
recall # 0.5420465
f1 <- 2 * ((precision * recall) / (precision + recall))
f1 # 0.7005603

# ROC - are under the curve a more viable metric for the accuracy of our model
# Both of the arguments on the ROC function need to be numeric!
class(tem.dataCSV.test$bugs)
dt.preds <- as.numeric(svm.predict)
roc.accuracy <- roc(tem.dataCSV.test$bugs, dt.preds)
print(roc.accuracy)
plot(roc.accuracy)
# The area under the curve is 0.7189

################################## Tune the SVM ################################
# find optimal cost of misclassification
tune.out <- tune(method = svm,
                 formula = bugs ~., 
                 data = train.smote.both, 
                 kernel = "linear",
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
# extract the best model

tune.out$optimal

(valid <- table(true = train.smote.both,
                pred = predict(tune.out$optimal, newx = train.smote.both)))

plot(svm.model, train.smote.both, decision.v) # adaptable to other dimensions


