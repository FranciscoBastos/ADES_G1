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
# tem.dataCSV <- dataCSV[1:1000,-c(1:7,ncol(dataCSV))]
tem.dataCSV <- dataCSV[1:25000,-c(1:7)]

tem.dataCSV$bugs <- as.integer(as.logical(tem.dataCSV$bugs))

# K-Means algorithm from the simplified data-set.
kmeans.model <- kmeans(tem.dataCSV, centers=2)

##############################Visualize our data################################
# Inspired by the blog post:
# https://towardsdatascience.com/how-to-create-a-correlation-matrix-with-too-many-variables-309cc0c0a57
#
# TODO Change this function to see what variables
# have more correlation with bugs!!
################################################################################

corr_simple <- function(data=tem.dataCSV, sig=0.95){
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
  
  # Plot correlations visually
  corrplot(mtx_corr, is.corr=FALSE, tl.col="black", na.label=" ")
}

corr_simple()

#################################Divide our data################################

set.seed(2987465)
index <- sample(1:nrow(tem.dataCSV), as.integer(0.7*nrow(tem.dataCSV)))
tem.dataCSV.train <- tem.dataCSV[index,]
tem.dataCSV.test <- tem.dataCSV[-index,]

dim(tem.dataCSV.train)
dim(tem.dataCSV.test)

#################################Decision tree##################################
# dev.new(width=15,height=14,noRStudioGD = TRUE)
png(file = "decision_tree.png", width = 4524, height = 900, units = "px")

output.tree <- ctree(
  bugs ~ ., 
  data = tem.dataCSV.train)

print(output.tree)

plot(output.tree)
dev.off()

# Prediction
predict.bug <- predict(output.tree, tem.dataCSV.test, type = 'prob')
table_bug <- table(tem.dataCSV.test$bugs, predict.bug)
table_bug

# Confusion matrix 
m.conf<-table(tem.dataCSV.test$bugs, predict.bug)

print(m.conf)

accuracy <- sum( diag(m.conf)) / sum (m.conf)

print(accuracy)

# Train the model with different samples to see if the accuracy changes
set.seed(2987465)

accuracy <- c()

for (i in 1:10) {
  index <- sample(1:nrow(tem.dataCSV), 0.7 * nrow(tem.dataCSV))
  
  data_train <- tem.dataCSV[index,]
  data_test <- tem.dataCSV[-index,]
  
  rpart_model <- ctree(
    bugs ~ ., 
    data = tem.dataCSV.train)
  
  rpart_pred <- predict(rpart_model, data_test, type = "prob")
  
  m.conf <- table(data_test$bugs, rpart_pred)
  
  accuracy <- c(accuracy, sum( diag(m.conf)) / sum (m.conf))
}

mean <- mean(accuracy)
sd <- sd(accuracy)

sprintf("%f is the mean", mean)
sprintf("%f is the standard deviation", sd)

# Evaluation metrics

modelEvaluation <- function(test, prediction) {
  if(length(unique(test)) == length(unique(prediction))){
    m.conf <- table(test, prediction)
    
    accuracy <- sum(diag(m.conf))/sum(m.conf)
    precision <- m.conf[1, 1]/sum(m.conf[,1])
    recall <- m.conf[1, 1]/sum(m.conf[1,])
    f1 <- 2 * precision * recall / (precision - recall)
    
    return (
      data.frame(
        accuracy <- round(accuracy, digits = 3),
        precision <- round(precision, digits = 3),
        recall <- round(recall, digits = 3),
        f1 <- round(f1, digits = 3)
      )
    )
    
  }
}

modelEvaluation(tem.dataCSV.test, predict.bug)
