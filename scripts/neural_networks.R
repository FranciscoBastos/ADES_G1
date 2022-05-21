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
################################### Both ######################################
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

train.smote.both.target <- tem.dataCSV.train.SMOTE

# Visualize the data
barplot(prop.table(table(train.smote.both$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class distribution (SMOTE, Over, Under)")

##################### Apply the neural networks algorithm ######################
# Verify if the all the variables are factors because the neural networks
# algorithms only work with numeric values
str(train.smote.both, list.len=ncol(train.smote.both))
# Change all variables to numeric
train.smote.both %<>% mutate_if(is.factor, as.numeric)
str(tem.dataCSV.test, list.len=ncol(tem.dataCSV.test))
# Change all variables to numeric
tem.dataCSV.test %<>% mutate_if(is.integer, as.numeric)
###################### Normalize the training data set #########################
# https://www.geeksforgeeks.org/how-to-normalize-and-standardize-data-in-r/
# Custom function to implement min max scaling

train.smote.both.norm <- scale(train.smote.both, scale = TRUE)

########################### Preliminary setup ##################################
# Start at 10 nodes and iterate 5 to 5 until 100,
# With 2 hidden layers maximum
################################################################################
neural.net.bugs <- neuralnet(bugs ~ NOI + RFC + CBO + WMC + 
                               Coupling.Metric.Rules + JUnit.Rules + 
                               Strict.Exception.Rules + NII + CBOI + LLOC +
                               TLLOC + NA. + NOA + TNOS + NLE + TLOC + 
                               Complexity.Metric.Rules + LOC + DIT +
                               NL + NOS + WarningMajor + WarningMinor + 
                               Unnecessary.and.Unused.Code.Rules + 
                               WarningInfo + TNA + NLM + 
                               Type.Resolution.Rules + Clone.Metric.Rules +
                               PUA + NM + Documentation.Metric.Rules +
                               Inheritance.Metric.Rules + LDC + NLA +
                               LLDC + NG + TNLS + CI + Brace.Rules + 
                               String.and.StringBuffer.Rules + CD +
                               Cohesion.Metric.Rules + AD + Android.Rules +
                               Basic.Rules + CC + CCL + CCO + CLC + 
                               CLLC + CLOC + Clone.Implementation.Rules +
                               Controversial.Rules + Design.Rules +
                               DLOC + Empty.Code.Rules +
                               Finalizer.Rules + Import.Statement.Rules +
                               J2EE.Rules,
                             data = train.smote.both.norm,
                             hidden = c(10, 3),
                             linear.output = FALSE,
                             lifesign = 'full',
                             rep = 3)
########################## Plot for better data visualization ##################
plot(neural.net.bugs, col.hidden = 'darkgreen',
     col.hidden.synapse = 'darkgreen', 
     show.weights = F,
     information = F,
     fill = 'lightblue')
#################################### Scaling ###################################
m <- colMeans(train.smote.both)
s <- apply(train.smote.both, 2, sd)
train.smote.both <- scale(train.smote.both, center = m, scale = s)
tem.dataCSV.test <- scale(tem.dataCSV.test, center = m, scale = s)
################################ Model creation ################################
neural.net.bugs <- keras_model_sequential()
neural.net.bugs %>%
  layer_dense(units = 5, activation = 'relu', input_shape = c(13)) %>%
  layer_dense(units = 1)
################################ Model compilation #############################
neural.net.bugs %>% compile(loss = 'mse', 
                            optimizer = optimizer_rmsprop(), 
                            metrics = 'mae')
############################### Model fitting ##################################
# TODO ERROR HERE FIX LATER
mymodel <- neural.net.bugs %>%          
  fit(train.smote.both, 
      train.smote.both.target,
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2)
################################# Predictions ##################################
neural.net.bugs$result.matrix
neural.net.bugs.preds <- compute(neural.net.bugs,
                                 rep = 1,
                                 tem.dataCSV.test)
p1 <- neural.net.bugs.preds$net.result
pred1 <- ifelse(p1 > 0.5, 1, 0)
pred1
cm.nn <- table(pred1, tem.dataCSV.test$bugs)
cm.nn
