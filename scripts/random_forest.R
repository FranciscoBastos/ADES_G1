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
install.packages("randomForest")


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
library("ggplot2")
library("RandomForestsGLS")


require(caTools)

suppressPackageStartupMessages(c(library(caret),library(corrplot),library(smotefamily)))

if(!require('DMwR')) {
  install.packages('DMwR')
  library('DMwR')
}

################################Load data#######################################

dataCSV <- read.csv("./data/dev.csv")

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

train.smote.both <- 
  ovun.sample(bugs~., data=tem.dataCSV.train.SMOTE, 
              method = "both")$data
table(train.smote.both$bugs)

# Visualize the data
barplot(prop.table(table(train.smote.both$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Bug class distribution (SMOTE, Over, Under sampeling)")

###################### Apply the random forest algorithm #######################
rf <- rfsrc(bugs ~ NOI + RFC + CBO + WMC + Coupling.Metric.Rules + 
              JUnit.Rules + Strict.Exception.Rules + NII + CBOI + LLOC +
              TLLOC + NA. + NOA + TNOS + NLE + TLOC + 
              Complexity.Metric.Rules + LOC + DIT + NL + NOS + 
              WarningMajor + WarningMinor + 
              Unnecessary.and.Unused.Code.Rules + WarningInfo + TNA + NLM + 
              Type.Resolution.Rules + Clone.Metric.Rules + PUA + NM + 
              Documentation.Metric.Rules + Inheritance.Metric.Rules + 
              LDC + NLA + LLDC + NG + TNLS + CI + Brace.Rules + 
              String.and.StringBuffer.Rules + CD + Cohesion.Metric.Rules + 
              AD + Android.Rules + Basic.Rules + CC + CCL + CCO + CLC + 
              CLLC + CLOC + Clone.Implementation.Rules +
              Controversial.Rules + Design.Rules + DLOC + Empty.Code.Rules +
              Finalizer.Rules + Import.Statement.Rules + J2EE.Rules,
            data = train.smote.both,
            method = "class")
rf
# Use the type = "class" for a classification tree!
rf.preds <- predict(rf, tem.dataCSV.test, type = "class")
rf.preds
# Use the type = "prob" to get the probabilities!
rf.pred.probs <- predict(rf, tem.dataCSV.test, type = "prob")
rf.pred.probs

################################################################################
#
# ATENTION: YOU NEED TO USE THE COMPETION DATA, THE ALL DATASET.
#           YOU CANNOT DEVIDE THE DATASET INTO TRAINING AND TESTING.
#           IT WILL NOT WORK, TRUST ME!
# SUBMISSION 1 AND 2:
#               RANDOM TEST SUBMISSIONS!
# 
# SUBMISSION 3: 
#               THE ROC ON THE THIRD TRY WAS OF 0.618431 
#               WITH THE CLASSIFICATION TREE!
# SUBMISSION 4:
#               THE ROC ON THE FORTH TRY WAS OF 0.7349
#               WITH THE CLASSIFICATION TREE AND BOTH SAMPELING!
# SUBMISSION 5:
#               THE ROC ON THE FIFHT TRY WAS OF 0.7349
#               WITH THE CLASSIFICATION TREE AND SOMTE SAMPELING!
# SUBMISSION 6:
#               THE ROC ON THE SIXTH TRY WAS OF 0.7228
#               WITH THE CLASSIFICATION TREE, SOMTE SAMPELING 
#               AND ALSO THE VARIABLES WITH HIGHER CORELATION (70 percent)!
# SUBMISSION 7:
#               THE ROC ON THE SEVENTH TRY WAS OF 0.7277
#               WITH THE CLASSIFICATION TREE, OVER AND UNDER SAMPELING, 
#               AS WELL AS, WITH SOMTE,
#               AND ALSO WITH ALL THE VARIABLES!
#
# SUBMISSION 8:
#               THE ROC ON THE EIGHT TRY WAS OF 0.8006
#               WITH THE LINEAR REGRESSION, OVER AND UNDER SAMPELING, 
#               AS WELL AS, WITH SOMTE,
#               AND ALSO WITH ONLY THE SELECTED VARIABLES!
# SUBMISSION 9:
#               THE ROC ON THE EIGHT TRY WAS OF 0.86713796
#               WITH THE RANDOM FOREST, OVER AND UNDER SAMPELING, 
#               AS WELL AS, WITH SOMTE,
#               AND ALSO WITH ONLY THE SELECTED VARIABLES!
#
# Export data set to more or less Kaggle format
################################################################################
dataCSVComp <- read.csv("data/comp.csv")
dataCSVComp <- na.omit(dataCSVComp)
tem.dataCSVComp <- subset(dataCSVComp, select=-c(ID,
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

#######################Decision tree with competition data######################

# Prediction
predict.bug <- predict(rf, tem.dataCSVComp, type = 'prob')
predict.bug
predict.bug$xvar
predict.bug$predicted

# Turn prediction data into a data.frame
df.predict.bug <- data.frame(predict.bug$xvar, predict.bug$predicted)
df.predict.bug

submission <- df.predict.bug[ , ncol(df.predict.bug), drop = FALSE] # apply ncol & drop
submission

write.csv(submission, "submissions/random_forest.csv")
