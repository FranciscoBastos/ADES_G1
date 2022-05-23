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

################################Load data#######################################

dataCSV <- read.csv("./data/dev.csv")
dataCSV <- na.omit(dataCSV)

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


##################### Apply the naive bayes algorithm ######################

model <- naive_bayes(bugs ~ ., data =  train.smote.both.target , usekernel = T) 
plot(model) 

p <- predict(model, train.smote.both.target, type = 'prob')
head(cbind(p, train.smote.both.target))


p1 <- predict(model, train.smote.both.target)
# Confusion Matrix

cm <- table( p1, train.smote.both.target$bugs)
cm

# Model Evaluation
confusionMatrix(cm)


misclassification_Traning <- 1-sum(diag(cm))/sum(cm)
misclassification_Traning

##Misclassification is around 22%%.

## Training model accuracy is around 78% !


#############Test data ###################3


p2 <- predict(model, tem.dataCSV.test)

cm2 <- table(p2, tem.dataCSV.test$bugs)
cm2
# Model Evaluation
confusionMatrix(cm2)

misclassification_Testing<-1-sum(diag(cm2))/sum(cm2)
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

# The area under the curve is 0.6726


################################################################################
dataCSVComp <- read.csv("data/comp.csv")
dataCSVComp <- na.omit(dataCSVComp)
tem.dataCSVComp <- dataCSVComp[-c(1:7)]

####################### Naive Bayes with competition data ######################
# Prediction
predict.bug <- predict(model,data = tem.dataCSVComp, type = 'prob')
predict.bug


# Save the data into submission format.
write.csv(predict.bug, "submissions/naive_bayes.csv")
