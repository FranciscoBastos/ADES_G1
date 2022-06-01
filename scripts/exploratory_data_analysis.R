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
library("randomForest")
library("doMC")
library("ISLR")
library("dlookr")


require(caTools)

suppressPackageStartupMessages(c(library(caret),
                                 library(corrplot),
                                 library(smotefamily)))

if(!require('DMwR2')) {
  install.packages('DMwR2')
  library('DMwR2')
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
# https://cran.r-project.org/web/packages/dlookr/vignettes/EDA.html
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
tem.dataCSV.balanced.sample <- tem.dataCSV.balanced[1:35000, ]
tem.dataCSV.balanced.sample <- na.omit(tem.dataCSV.balanced.sample)
tem.dataCSV.balanced.sample$MigratingToJUnit4.Rules # should be NULL
# Turn bugs logical column into a integer value of 1 or 0.
tem.dataCSV.balanced.sample$bugs <-
  as.integer(as.logical(tem.dataCSV.balanced.sample$bugs))

str(tem.dataCSV.balanced.sample)

describe.data <- describe(tem.dataCSV.balanced.sample)
write.csv(describe.data, "submissions/exploratory_data_analysis.csv")

bugs <- target_by(tem.dataCSV.balanced.sample, bugs)
bugs
summary(bugs)

tem.dataCSV.balanced.sample %>% eda_paged_report(target = "bugs", 
                                                 subtitle = "bugs", 
                                                 output_dir = "/Users/franciscobastos/Developer/ADES/ADES_G1/reports", 
                                                 output_file = "thebugcathcer.html",
                                                 theme = "blue")
