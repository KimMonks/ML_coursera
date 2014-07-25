library(ggplot2)
library(caret)
data(iris)
names(iris)
table(iris$Species)
inTrain <- createDataPartition(y=iris$Species,p=.7,list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training); dim(testing)
qplot(iris$Petal.Width,iris$Sepal.Width,colour=iris$Species)
modFit <- train(Species ~ .,method="rpart",data=training)
print(modFit$finalModel)
library(rattle)
fancyRpartPlot(modFit$finalModel)
