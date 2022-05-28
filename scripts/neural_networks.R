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
library("FeatureTerminatoR")
library("mlbench")
library("doMC")

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

###################### Normalize the training data set #########################
# https://www.geeksforgeeks.org/how-to-normalize-and-standardize-data-in-r/
# Custom function to implement min max scaling
# train.smote.both.norm <- scale(train.smote.both, scale = TRUE)
minmaxnorm <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

tem.dataCSV.balanced.sample <- 
  as.data.frame(lapply(tem.dataCSV.balanced.sample, minmaxnorm))

apply(tem.dataCSV.balanced.sample, 2, min)

apply(tem.dataCSV.balanced.sample, 2, max)

############################### Cleaning the data ##############################
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
        main = "Bugs class distribution")
############################### SMOTE function #################################
# Divide our balanced data 
# Trying with: SMOTE
#
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
X_train = tem.dataCSV.train[,-ncol(tem.dataCSV.balanced.sample)]
y_train = tem.dataCSV.train[,ncol(tem.dataCSV.balanced.sample)]

dim(tem.dataCSV.train)
dim(tem.dataCSV.test)
dim(X_train)
dim(y_train) # should be NULL

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
################################################################################
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
        main = "Bugs class distribution (SMOTE, Over, Under)")
##################### Apply the neural networks algorithm ######################
# Verify if the all the variables are factors because the neural networks
# algorithms only work with numeric values
str(train.smote.both, list.len=ncol(train.smote.both))
# Change all variables to numeric
train.smote.both %<>% mutate_if(is.factor, as.numeric)
str(tem.dataCSV.test, list.len=ncol(tem.dataCSV.test))
# Change all variables to numeric
tem.dataCSV.test %<>% mutate_if(is.integer, as.numeric)

# Classification ANNs in the neuralnet package 
# require that the response feature, in this case bugs, 
# be inputted as a Boolean feature.
# Train
# http://uc-r.github.io/ann_classification
class(train.smote.both$bugs)
train.smote.both <- train.smote.both %>%
  mutate(bugs = ifelse(bugs == 1, TRUE, FALSE))
class(train.smote.both$bugs)
str(train.smote.both)

# Classification ANNs in the neuralnet package 
# require that the response feature, in this case bugs, 
# be inputted as a Boolean feature.
# Test
# http://uc-r.github.io/ann_classification
class(tem.dataCSV.test$bugs)
tem.dataCSV.test <- tem.dataCSV.test %>%
  mutate(bugs = ifelse(bugs == 1, TRUE, FALSE))
class(tem.dataCSV.test$bugs)
str(tem.dataCSV.test)

########################### Preliminary setup ##################################
# Start at 10 nodes and iterate 5 to 5 until 100,
# With 2 hidden layers maximum
#
# This features are the most important 
# according with the bagged tree RFE selection
################################################################################
neural.net.bugs.first <- neuralnet(bugs ~ NOI + RFC + CBO + WMC + 
                                     Coupling.Metric.Rules + JUnit.Rules + 
                                     Strict.Exception.Rules + NII + CBOI + 
                                     LLOC + TLLOC + NA. + NOA + TNOS + NLE + 
                                     TLOC + Complexity.Metric.Rules +
                                     LOC + DIT + NL + NOS +
                                     WarningMajor + WarningMinor + 
                                     Unnecessary.and.Unused.Code.Rules +
                                     WarningInfo + TNA + NLM + 
                                     Type.Resolution.Rules +
                                     Clone.Metric.Rules + PUA + NM +
                                     Documentation.Metric.Rules +
                                     Inheritance.Metric.Rules + LDC + NLA +
                                     LLDC + NG + TNLS + CI + Brace.Rules +
                                     String.and.StringBuffer.Rules + CD +
                                     Cohesion.Metric.Rules + AD + 
                                     Android.Rules + Basic.Rules + CC +
                                     CCL + CCO + CLC + 
                                     CLLC + CLOC + Clone.Implementation.Rules +
                                     Controversial.Rules + Design.Rules +
                                     DLOC + Empty.Code.Rules +
                                     Finalizer.Rules + Import.Statement.Rules +
                                     J2EE.Rules,
                                   data = train.smote.both,
                                   linear.output = FALSE,
                                   err.fct = 'ce',
                                   likelihood = TRUE,
                                   stepmax=1e7)

########################## Plot for better data visualization ##################
plot(neural.net.bugs.first, rep = 'best')
############################ Compute the probability ###########################
output <- compute(neural.net.bugs.first, tem.dataCSV.test, rep = 1)
summary(output)
output$net.result # The probability

class(tem.dataCSV.test$bugs)
tem.dataCSV.test$bugs <- as.numeric(tem.dataCSV.test$bugs)
output <- as.numeric(output$net.result)
class(output)
roc.accuracy <- roc(tem.dataCSV.test$bugs, output)
print(roc.accuracy) # The area under the curve is 0.806
plot(roc.accuracy)

############################# Test for other models ############################
# Random Hyper Parameter Search
# Helps:
# https://topepo.github.io/caret/recursive-feature-elimination.html
# http://uc-r.github.io/ann_classification
# https://www.learnbymarketing.com/tutorials/neural-networks-in-r-tutorial/
# https://topepo.github.io/caret/random-hyperparameter-search.html 
# https://topepo.github.io/caret/train-models-by-tag.html#Neural_Network 
# https://stackoverflow.com/questions/44084735/classification-usage-of-factor-levels
# 
################################################################################
# configure multicore
registerDoMC(cores = 4)

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 1,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary,
                           search = "random")

set.seed(825)

# It needs or factor or numeric 
# Turn bugs logical column into a integer value of 1 or 0.
train.smote.both$bugs <-
  as.integer(as.logical(train.smote.both$bugs))
train.smote.both$bugs <- as.factor(train.smote.both$bugs)
class(train.smote.both$bugs)

neural.net.fit <- train(make.names(bugs) ~ ., 
                 data = train.smote.both, 
                 method = "nnet",
                 metric = "ROC",
                 tuneLength = 30, # 30 tuning parameter combinations
                 trControl = fitControl)
neural.net.fit$bestTune

ggplot(neural.net.fit) + theme(legend.position = "top")

################################################################################
# configure multicore
registerDoMC(cores = 4)

control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      number = 10,
                      verbose = TRUE)

subsets <- c(1:ncol(train.smote.both), 10, 15, 20, 25)
################################################################################
# configure multicore
registerDoMC(cores = 4)

neural.net.bugs.second <- neuralnet(bugs ~ NOI + RFC + CBO + WMC + 
                                     Coupling.Metric.Rules + JUnit.Rules + 
                                     Strict.Exception.Rules + NII + CBOI + 
                                     LLOC + TLLOC + NA. + NOA + TNOS + NLE + 
                                     TLOC + Complexity.Metric.Rules +
                                     LOC + DIT + NL + NOS +
                                     WarningMajor + WarningMinor + 
                                     Unnecessary.and.Unused.Code.Rules +
                                     WarningInfo + TNA + NLM + 
                                     Type.Resolution.Rules +
                                     Clone.Metric.Rules + PUA + NM +
                                     Documentation.Metric.Rules +
                                     Inheritance.Metric.Rules + LDC + NLA +
                                     LLDC + NG + TNLS + CI + Brace.Rules +
                                     String.and.StringBuffer.Rules + CD +
                                     Cohesion.Metric.Rules + AD + 
                                     Android.Rules + Basic.Rules + CC +
                                     CCL + CCO + CLC + 
                                     CLLC + CLOC + Clone.Implementation.Rules +
                                     Controversial.Rules + Design.Rules +
                                     DLOC + Empty.Code.Rules +
                                     Finalizer.Rules + Import.Statement.Rules +
                                     J2EE.Rules,
                                   data = train.smote.both,
                                   linear.output = FALSE,
                                   err.fct = 'ce',
                                   likelihood = TRUE,
                                   stepmax = 1e7,
                                   hidden = c(9, 7))

# TODO MAKE A PREDICTION WITH THE COMPETION DATA
