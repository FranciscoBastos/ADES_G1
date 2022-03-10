library("psych")
library("corrplot")
library("rpart")
library("neuralnet")

################################Load data#######################################

dataCSV <- read.csv("data/dev.csv")

############################Primary analysis####################################

summary(dataCSV)
head(dataCSV)
tail(dataCSV)
str(dataCSV)
dim(dataCSV)





