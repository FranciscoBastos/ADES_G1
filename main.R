library("psych")
library("corrplot")
library("rpart")
library("neuralnet")
library("dplyr")

################################Load data#######################################

dataCSV <- read.csv("data/dev.csv")

##############################Primary analysis##################################

summary(dataCSV)
head(dataCSV)
tail(dataCSV)
str(dataCSV)
dim(dataCSV)

set.seed(2987465)
dataCSV <- na.omit(dataCSV)
dataCSV <- as.numeric(dataCSV)

dataCSV[sample.int(nrow(dataCSV), 1000), ]

############################K-Means clustering algorithm########################

# Create a vector for analysis with only the first 1000 columns of the data-set.
# Also we removed the 1 to the 3 columns and the last - because they had 
# non numerical values. 
tem.dataCSV <- dataCSV[1:1000,-c(1:3,ncol(dataCSV))]

# K-Means algorithm from the simplified dataset.
kmeans(tem.dataCSV, centers=2)




