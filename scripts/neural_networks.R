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
library("randomForest")
library("keras")
library("mlbench")
library("magrittr")

install_tensorflow()
use_condaenv("r-tensorflow")

require(caTools)

suppressPackageStartupMessages(c(library(caret),library(corrplot),library(smotefamily)))

if(!require('DMwR')) {
  install.packages('DMwR')
  library('DMwR')
}

################################Load data#######################################

dataCSV <- read.csv("./data/dev.csv")
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
##################### Analyze our data set and fix unbalanced ##################
# Re-sample the data in 55000 samples
tem.dataCSV.balanced.sample <- dataCSV[1:55000, -c(1:7)]
tem.dataCSV.balanced.sample <- na.omit(tem.dataCSV.balanced.sample)
# Turn bugs logical column into a integer value of 1 or 0.
tem.dataCSV.balanced.sample$bugs <-
  as.integer(as.logical(tem.dataCSV.balanced.sample$bugs))

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
neural.net.bugs <- neuralnet(bugs ~ CC + CCL + CCO + CI + CLC + CLLC + LDC + 
                               LLDC + LCOM5 + NL + NLE + WMC + CBO + CBOI +
                               NII + NOI + RFC + AD + CD + CLOC + DLOC + PDA +
                               PUA + TCD + TCLOC + DIT + NOA + NOC + NOD + NOP + 
                               LLOC + LOC + NA.  + NG + NLA + NLG + NLM  +
                               NLPA + NLPM + NLS + NM + NOS + NPA + NPM +
                               NS + TLLOC + TLOC + TNA + TNG  + TNLA + TNLG +
                               TNLM + TNLPA + TNLPM + TNLS + TNM + TNOS +
                               TNPA + TNPM + TNS + WarningBlocker + 
                               WarningCritical + WarningInfo + WarningMajor +
                               WarningMinor + Android.Rules + Basic.Rules + 
                               Brace.Rules + Clone.Implementation.Rules +
                               Clone.Metric.Rules + Code.Size.Rules +
                               Cohesion.Metric.Rules + Comment.Rules +
                               Complexity.Metric.Rules + Controversial.Rules +
                               Coupling.Metric.Rules + Coupling.Rules +
                               Design.Rules + Documentation.Metric.Rules +
                               Empty.Code.Rules + Finalizer.Rules +
                               Import.Statement.Rules + 
                               Inheritance.Metric.Rules + J2EE.Rules +
                               JUnit.Rules + Jakarta.Commons.Logging.Rules +
                               Java.Logging.Rules + Migration13.Rules +
                               Migration14.Rules + Migration15.Rules +
                               JavaBean.Rules + MigratingToJUnit4.Rules +
                               Migration.Rules + Naming.Rules +
                               Optimization.Rules + Type.Resolution.Rules +
                               Unnecessary.and.Unused.Code.Rules + 
                               Vulnerability.Rules + 
                               Security.Code.Guideline.Rules + 
                               Size.Metric.Rules + Strict.Exception.Rules + 
                               String.and.StringBuffer.Rules,
                             data = train.smote.both.norm,
                             hidden = c(12,7), # Start at 10 - 5 - 5 - 100, with 2 hidden layers.
                             linear.output = FALSE,
                             lifesign = 'full',
                             rep = 5)
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
