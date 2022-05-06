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
tem.dataCSV.balanced.sample <- dataCSV[1:55000, -c(1:7)]
tem.dataCSV.balanced.sample <- na.omit(tem.dataCSV.balanced.sample)
# Turn bugs logical column into a integer value of 1 or 0.
tem.dataCSV.balanced.sample$bugs <-
  as.integer(as.logical(tem.dataCSV.balanced.sample$bugs))

suppressMessages(library(plm)) # to remove note about dependencies
is.pbalanced(tem.dataCSV.balanced.sample)
# Is the data set balanced?
# $ FALSE
# The data set is not balanced!

suppressMessages(library(tidyverse))

tem.dataCSV.balanced <- make.pbalanced(tem.dataCSV.balanced.sample,
                                       balance.type = c("fill"))

class(tem.dataCSV.balanced$bugs)

# Turn bugs logical column into a integer value of 1 or 0.
tem.dataCSV.balanced$bugs <-
  as.integer(as.logical(tem.dataCSV.balanced$bugs))

is.pbalanced(tem.dataCSV.balanced)
# Is the data set balanced?
# $ TRUE
# The data set is balanced!

table(tem.dataCSV.balanced.sample$bugs)
# Are the bugs balanced ?
# No the bugs are not balanced.
# Let's balance our bugs.

# Visualize the data
barplot(prop.table(table(tem.dataCSV.balanced.sample$bugs)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class distribution")

##############################Visualize our data################################
# Inspired by the blog post:
# https://towardsdatascience.com/how-to-create-a-correlation-matrix-with-too-many-variables-309cc0c0a57
#
################################################################################
# Usually a correlation of 70% is pretty high, however due to the characteristic 
# of our data set it is advisable to use a correlation of 95%
################################################################################

corr_simple <- function(data=tem.dataCSV.balanced.sample, sig=0.95){
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
        main = "Class distribution (SMOTE, Over, Under)")

##################### Linear regression - selection of variables ###############
# https://quantifyinghealth.com/stepwise-selection/
# https://www.kaggle.com/code/mahmoud86/tutorial-subset-selection-methods/notebook
# https://towardsdatascience.com/selecting-the-best-predictors-for-linear-regression-in-r-f385bf3d93e9
#
################################################################################

library("leaps")
subset <-
  regsubsets(bugs~.,
             data =train.smote.both,
             nbest = 1,      # 1 best model for each number of predictors
             nvmax = 60,    # 60 for no limit on number of variables
             force.in = NULL, force.out = NULL,
             method = "backward",
             really.big = TRUE)
summary_best_subset <- summary(subset)
as.data.frame(summary_best_subset$outmat)
which.max(summary_best_subset$adjr2)
summary_best_subset$which[61,]

class(train.smote.both$bugs)

train.smote.both$bugs <- as.numeric(train.smote.both$bugs)

lm.model <- lm(bugs ~  
                 CC + CCL + CCO + CI + CLC + CLLC + LDC + LLDC + LCOM5 + NL + 
                 NLE + WMC + CBO + CBOI + NII + NOI + RFC + AD + CD + CLOC + 
                 DLOC + PDA + PUA + TCD + TCLOC + DIT + NOA + NOC + NOD + NOP + 
                 LLOC + LOC + NA.  + NG + NLA + NLG + NLM  + NLPA + NLPM + NLS +
                 NM + NOS + NPA + NPM + NS + TLLOC + TLOC + TNA + TNG  + TNLA + 
                 TNLG  + TNLM + TNLPA + TNLPM + TNLS + TNM + TNOS + TNPA+ TNPM +
                 TNS + WarningBlocker + WarningCritical + WarningInfo + 
                 WarningMajor + WarningMinor + Android.Rules + Basic.Rules + 
                 Brace.Rules + Clone.Implementation.Rules + Clone.Metric.Rules +
                 Code.Size.Rules + Cohesion.Metric.Rules + Comment.Rules +
                 Complexity.Metric.Rules + Controversial.Rules +
                 Coupling.Metric.Rules + Coupling.Rules + Design.Rules + 
                 Documentation.Metric.Rules + Empty.Code.Rules +
                 Finalizer.Rules + Import.Statement.Rules +
                 Inheritance.Metric.Rules + J2EE.Rules + JUnit.Rules + 
                 Jakarta.Commons.Logging.Rules + Java.Logging.Rules +
                 JavaBean.Rules + MigratingToJUnit4.Rules + Migration.Rules +
                 Migration13.Rules + Migration14.Rules + Migration15.Rules +
                 Naming.Rules + Optimization.Rules +
                 Type.Resolution.Rules + Unnecessary.and.Unused.Code.Rules +
                 Vulnerability.Rules + Security.Code.Guideline.Rules + 
                 Size.Metric.Rules + Strict.Exception.Rules + 
                 String.and.StringBuffer.Rules, data = train.smote.both)

lm.pred <- predict(lm.model, tem.dataCSV.test)
summary(lm.pred)

# Model statistics
class(tem.dataCSV.test$bugs)
lm.pred <- as.numeric(lm.pred)
class(lm.pred)
roc.accuracy <- roc(tem.dataCSV.test$bugs, lm.pred)
print(roc.accuracy) # 0.8006
plot(roc.accuracy)

lm.mad <- sum(abs(lm.pred - tem.dataCSV.test$bugs))/length(lm.pred)
lm.mad # 1.329709
lm.mse <- sum((lm.pred - tem.dataCSV.test$bugs)^2)/length(lm.pred)
lm.mse # 1.83478

d <- tem.dataCSV.test$bugs - lm.pred
lm.mae <- mean(abs(d))
lm.mae # 1.329709
lm.rmse <- sqrt(mean(d^2))
lm.rmse # 1.354541

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
#
# Export data set to more or less Kaggle format
################################################################################
dataCSVComp <- read.csv("data/comp.csv")
dataCSVComp <- na.omit(dataCSVComp)
tem.dataCSVComp <- dataCSVComp[-c(1:7)]

#######################Decision tree with competition data######################

# Prediction
predict.bug <- predict(lm.model, tem.dataCSVComp)
predict.bug

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
