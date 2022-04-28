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

##############################Visualize our data################################
# Inspired by the blog post:
# https://towardsdatascience.com/how-to-create-a-correlation-matrix-with-too-many-variables-309cc0c0a57
#
################################################################################

corr_simple <- function(data=balanced.data.both.sampling, sig=0.70){
        # Convert data to numeric in order to run correlations
        # Convert to factor first to keep the integrity of the data - 
        # Each value will become a number rather than turn into NA
        df_cor <- data %>% mutate_if(is.character, as.factor)
        df_cor <- df_cor %>% mutate_if(is.factor, as.numeric)
        
        # Run a correlation and drop the insignificant ones
        corr <- cor(df_cor)
        
        # Prepare to drop duplicates and correlations of 1
        corr[lower.tri(corr,diag=TRUE)] <- NA 
        # Drop perfect correlations
        corr[corr == 1] <- NA 
        
        # Turn into a 3-column table
        corr <- as.data.frame(as.table(corr))
        # Remove the NA values from above 
        corr <- na.omit(corr) 
        
        # Select significant values  
        corr <- subset(corr, abs(Freq) > sig) 
        
        # Sort by highest correlation
        corr <- corr[order(-abs(corr$Freq)),] 
        # Print table
        print(corr)
        
        # Turn corr back into matrix in order to plot with corrplot
        mtx_corr <- reshape2::acast(corr, Var1~Var2, value.var="Freq")
        
        png(file="plots/corr.png", res=300, width=4500, height=4500)
        
        # Plot correlations visually
        corrplot(mtx_corr, is.corr=FALSE, tl.col="black", na.label=" ")
        
        dev.off()
}

corr_simple()

################Trying with only the variables with correlation ################
dt <- rpart( bugs ~ 
                     CC + CCL + CCO + CI + CLC + CLLC + LDC + NL + 
                     CBO + CBOI + NOI + RFC + AD + CLOC + DLOC + PUA + TCLOC +
                     DIT + NOA + LLOC + LOC + NG + NLA + NLG + NLPA + 
                     NLPM + NLS + NM + NOS + NPA + NPM + NS + TLLOC + TLOC + 
                     TNA + TNG + TNLA + TNLG + TNLM + TNLPA + TNLPM + TNLS + 
                     TNM + TNOS + TNPM + WarningMajor + WarningMinor + 
                     Documentation.Metric.Rules + CD,
             data = tem.dataCSV.train.SMOTE,
             method = "class")
dt
# Use the type = "class" for a classification tree!
dt.preds <- predict(dt, tem.dataCSV.test, type = "class")
dt.preds
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
# The area under the curve is 0.7228
# Other statistics
# inspired by: https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9 
precision <- cm.dt[1, 1]/sum(cm.dt[,1])
precision
recall <- cm.dt[1, 1]/sum(cm.dt[1,])
recall
f1 <- 2 * ((precision * recall) / (precision + recall))
f1

################Trying with only the variables with correlation ################
########################Searching for over-fitting again #######################
dt <- rpart( bugs ~ 
                     CC + CCL + CCO + CI + CLC + CLLC + LDC + NL + 
                     CBO + CBOI + NOI + RFC + AD + CLOC + DLOC + PUA + TCLOC +
                     DIT + NOA + LLOC + LOC + NG + NLA + NLG + NLPA + 
                     NLPM + NLS + NM + NOS + NPA + NPM + NS + TLLOC + TLOC + 
                     TNA + TNG + TNLA + TNLG + TNLM + TNLPA + TNLPM + TNLS + 
                     TNM + TNOS + TNPM + WarningMajor + WarningMinor + 
                     Documentation.Metric.Rules + CD,
             data = tem.dataCSV.train.SMOTE,
             method = "class")
dt
############################ For training dataset ##############################
# Use the type = "class" for a classification tree!
dt.preds <- predict(dt, tem.dataCSV.train.SMOTE, type = "class")
dt.preds
# Use the type = "prob" to get the probabilities!
dt.pred.probs <- predict(dt, tem.dataCSV.train.SMOTE, type = "prob")
dt.pred.probs

# compute confusion matrix for training data
cm.dt <- table(dt.preds, tem.dataCSV.train.SMOTE$bugs)
cm.dt

accuracy <- sum(diag(cm.dt)) / sum(cm.dt)
accuracy # 0.8270894
error.dt.test <- 1 - sum(diag(cm.dt)) / sum(cm.dt)
error.dt.test # 0.1729106

class(tem.dataCSV.train.SMOTE$bugs)
dt.preds <- as.numeric(dt.preds)
class(dt.preds)
roc.accuracy <- roc(tem.dataCSV.train.SMOTE$bugs, dt.preds)
print(roc.accuracy)
plot(roc.accuracy)
# The area under the curve is 0.8271
# Other statistics
# inspired by: https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9 
precision <- cm.dt[1, 1]/sum(cm.dt[,1])
precision # 0.807824
recall <- cm.dt[1, 1]/sum(cm.dt[1,])
recall # 0.8406463
f1 <- 2 * ((precision * recall) / (precision + recall))
f1 # 0.8239084
############################### For test dataset ###############################
# Use the type = "class" for a classification tree!
dt.preds <- predict(dt, tem.dataCSV.test, type = "class")
dt.preds
# Use the type = "prob" to get the probabilities!
dt.pred.probs <- predict(dt, tem.dataCSV.test, type = "prob")
dt.pred.probs

# compute confusion matrix
cm.dt <- table(dt.preds, tem.dataCSV.test$bugs)
cm.dt

accuracy <- sum(diag(cm.dt)) / sum(cm.dt)
accuracy # 0.8094262
error.dt.test <- 1 - sum(diag(cm.dt)) / sum(cm.dt)
error.dt.test # 0.1905738

class(tem.dataCSV.test$bugs)
dt.preds <- as.numeric(dt.preds)
class(dt.preds)
roc.accuracy <- roc(tem.dataCSV.test$bugs, dt.preds)
print(roc.accuracy)
plot(roc.accuracy)
# The area under the curve is 0.6939
# Other statistics
# inspired by: https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9 
precision <- cm.dt[1, 1]/sum(cm.dt[,1])
precision # 0.8194593
recall <- cm.dt[1, 1]/sum(cm.dt[1,])
recall # 0.9785472
f1 <- 2 * ((precision * recall) / (precision + recall))
f1 # 0.8919652
################Trying with only the variables with correlation ################
# According to the post: 
# https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/
# We can use SOMTE with under sampling to obtain better results!
# Trying: CORRELATION VARIABLES + SMOTE + UNDERSAMPLING
#
################################################################################
################Trying with only the variables with correlation ################
set.seed(2987465)
index <- sample(1:nrow(tem.dataCSV.balanced.sample), 
                as.integer(0.7*nrow(tem.dataCSV.balanced.sample)))
tem.dataCSV.train <- tem.dataCSV.balanced.sample[index,]
tem.dataCSV.test <- tem.dataCSV.balanced.sample[-index,]

dim(tem.dataCSV.train)
dim(tem.dataCSV.test)

balanced.data.under.sampling.SOMTE.train <- 
        ovun.sample(bugs~., 
                    data=tem.dataCSV.train, 
                    method = "under")$data
table(balanced.data.under.sampling.SOMTE.train$bugs)

# Visualize the data
barplot(prop.table(table(balanced.data.under.sampling.SOMTE.train$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution under sampling")

# The SMOTE function requires the target variable to be numeric
tem.dataCSV.train$bugs <- as.numeric(tem.dataCSV.train$bugs)
tem.dataCSV.test$bugs <- as.numeric(tem.dataCSV.test$bugs)
class(tem.dataCSV.train$bugs)
class(tem.dataCSV.test$bugs)
# It is a numeric now!

# For the training data set
# All but the last column
tem.dataCSV.train.under.SMOTE <- 
        SMOTE(balanced.data.under.sampling.SOMTE.train[,-ncol(balanced.data.under.sampling.SOMTE.train)],
              balanced.data.under.sampling.SOMTE.train$bugs,
              K = 5)

# Extract only the balanced dataset
tem.dataCSV.train.under.SMOTE <- tem.dataCSV.train.under.SMOTE$data
# Change the name from class to bugs
colnames(tem.dataCSV.train.under.SMOTE) [ncol(tem.dataCSV.train.under.SMOTE)] <- "bugs"
tem.dataCSV.train.under.SMOTE$bugs <- as.factor(tem.dataCSV.train.under.SMOTE$bugs)
table(tem.dataCSV.train.under.SMOTE$bugs)

# Visualize the data
barplot(prop.table(table(tem.dataCSV.train.under.SMOTE$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution under sampling and SMOTE")

dt <- rpart( bugs ~ 
                     CC + CCL + CCO + CI + CLC + CLLC + LDC + NL + 
                     CBO + CBOI + NOI + RFC + AD + CLOC + DLOC + PUA + TCLOC +
                     DIT + NOA + LLOC + LOC + NG + NLA + NLG + NLPA + 
                     NLPM + NLS + NM + NOS + NPA + NPM + NS + TLLOC + TLOC + 
                     TNA + TNG + TNLA + TNLG + TNLM + TNLPA + TNLPM + TNLS + 
                     TNM + TNOS + TNPM + WarningMajor + WarningMinor + 
                     Documentation.Metric.Rules + CD,
             data = tem.dataCSV.train.under.SMOTE,
             method = "class")
dt

# Use the type = "class" for a classification tree!
dt.preds <- predict(dt, tem.dataCSV.test, type = "class")
dt.preds
# Use the type = "prob" to get the probabilities!
dt.pred.probs <- predict(dt, tem.dataCSV.test, type = "prob")
dt.pred.probs

# compute confusion matrix and statistcs
cm.dt <- table(dt.preds, tem.dataCSV.test$bugs)
cm.dt
accuracy <- sum(diag(cm.dt)) / sum(cm.dt)
accuracy # 0.653347
error.dt.test <- 1 - sum(diag(cm.dt)) / sum(cm.dt)
error.dt.test # 0.346653
class(tem.dataCSV.test$bugs)
dt.preds <- as.numeric(dt.preds)
class(dt.preds)
roc.accuracy <- roc(tem.dataCSV.test$bugs, dt.preds)
print(roc.accuracy) # 0.7007
plot(roc.accuracy)
precision <- cm.dt[1, 1]/sum(cm.dt[,1])
precision # 0.6492351
recall <- cm.dt[1, 1]/sum(cm.dt[1,])
recall # 0.9843581
f1 <- 2 * ((precision * recall) / (precision + recall))
f1 # 0.7824223

############################ BOTH + SMOTE + CORRELATION ########################
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

# For the training data set
# All but the last column
tem.dataCSV.train.both.SMOTE <- 
        SMOTE(balanced.data.both.sampling[,-ncol(balanced.data.both.sampling)],
              balanced.data.both.sampling$bugs,
              K = 5)

# Extract only the balanced dataset
tem.dataCSV.train.both.SMOTE <- tem.dataCSV.train.both.SMOTE$data
# Change the name from class to bugs
colnames(tem.dataCSV.train.both.SMOTE) [ncol(tem.dataCSV.train.both.SMOTE)] <- "bugs"
tem.dataCSV.train.both.SMOTE$bugs <- as.factor(tem.dataCSV.train.both.SMOTE$bugs)
table(tem.dataCSV.train.both.SMOTE$bugs)

# Visualize the data
barplot(prop.table(table(tem.dataCSV.train.both.SMOTE$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution under + ovrer sampling and SMOTE")

dt <- rpart( bugs ~ 
                     CC + CCL + CCO + CI + CLC + CLLC + LDC + NL + 
                     CBO + CBOI + NOI + RFC + AD + CLOC + DLOC + PUA + TCLOC +
                     DIT + NOA + LLOC + LOC + NG + NLA + NLG + NLPA + 
                     NLPM + NLS + NM + NOS + NPA + NPM + NS + TLLOC + TLOC + 
                     TNA + TNG + TNLA + TNLG + TNLM + TNLPA + TNLPM + TNLS + 
                     TNM + TNOS + TNPM + WarningMajor + WarningMinor + 
                     Documentation.Metric.Rules + CD,
             data = tem.dataCSV.train.both.SMOTE,
             method = "class")
dt

# Use the type = "class" for a classification tree!
dt.preds <- predict(dt, tem.dataCSV.test, type = "class")
dt.preds
# Use the type = "prob" to get the probabilities!
dt.pred.probs <- predict(dt, tem.dataCSV.test, type = "prob")
dt.pred.probs

# compute confusion matrix and statistcs
cm.dt <- table(dt.preds, tem.dataCSV.test$bugs)
cm.dt
accuracy <- sum(diag(cm.dt)) / sum(cm.dt)
accuracy # 0.704235
error.dt.test <- 1 - sum(diag(cm.dt)) / sum(cm.dt)
error.dt.test # 0.295765
class(tem.dataCSV.test$bugs)
dt.preds <- as.numeric(dt.preds)
class(dt.preds)
roc.accuracy <- roc(tem.dataCSV.test$bugs, dt.preds)
print(roc.accuracy) # 0.7067
plot(roc.accuracy)
precision <- cm.dt[1, 1]/sum(cm.dt[,1])
precision # 0.7040199
recall <- cm.dt[1, 1]/sum(cm.dt[1,])
recall # 0.9831098
f1 <- 2 * ((precision * recall) / (precision + recall))
f1 # 0.8204809

############################ BOTH + SMOTE + ALL ########################
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

# For the training data set
# All but the last column
tem.dataCSV.train.both.SMOTE <- 
        SMOTE(balanced.data.both.sampling[,-ncol(balanced.data.both.sampling)],
              balanced.data.both.sampling$bugs,
              K = 5)

# Extract only the balanced dataset
tem.dataCSV.train.both.SMOTE <- tem.dataCSV.train.both.SMOTE$data
# Change the name from class to bugs
colnames(tem.dataCSV.train.both.SMOTE) [ncol(tem.dataCSV.train.both.SMOTE)] <- "bugs"
tem.dataCSV.train.both.SMOTE$bugs <- as.factor(tem.dataCSV.train.both.SMOTE$bugs)
table(tem.dataCSV.train.both.SMOTE$bugs)

# Visualize the data
barplot(prop.table(table(tem.dataCSV.train.both.SMOTE$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution under + ovrer sampling and SMOTE")

dt <- rpart( bugs ~ .,
             data = tem.dataCSV.train.both.SMOTE,
             method = "class")
dt

# Use the type = "class" for a classification tree!
dt.preds <- predict(dt, tem.dataCSV.test, type = "class")
dt.preds
# Use the type = "prob" to get the probabilities!
dt.pred.probs <- predict(dt, tem.dataCSV.test, type = "prob")
dt.pred.probs

# compute confusion matrix and statistcs
cm.dt <- table(dt.preds, tem.dataCSV.test$bugs)
cm.dt
accuracy <- sum(diag(cm.dt)) / sum(cm.dt)
accuracy # 0.7563183
error.dt.test <- 1 - sum(diag(cm.dt)) / sum(cm.dt)
error.dt.test # 0.2436817
class(tem.dataCSV.test$bugs)
dt.preds <- as.numeric(dt.preds)
class(dt.preds)
roc.accuracy <- roc(tem.dataCSV.test$bugs, dt.preds)
print(roc.accuracy) # 0.7277
plot(roc.accuracy)
precision <- cm.dt[1, 1]/sum(cm.dt[,1])
precision # 0.7588047
recall <- cm.dt[1, 1]/sum(cm.dt[1,])
recall # 0.9836292
f1 <- 2 * ((precision * recall) / (precision + recall))
f1 # 0.8567125

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
# Export data set to more or less Kaggle format
################################################################################
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

