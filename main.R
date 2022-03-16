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
# For more information, if there is an error go-to:
# https://data-hacks.com/r-error-do_one-nmeth-na-nan-inf-foreign-function-call-arg-1
################################################################################
# Create a vector for analysis with only the first 1000 columns of the data-set.
# Also we removed the 1 to the 3 columns and the last - because they had 
# non numerical values. 
tem.dataCSV <- dataCSV[1:1000,-c(1:3,ncol(dataCSV))]

# K-Means algorithm from the simplified data-set.
kmeans(tem.dataCSV, centers=2)




