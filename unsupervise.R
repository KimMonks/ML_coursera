data(iris)
inTrain <- createDataPartition(y=iris$Species,p=.7,list=F)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training); dim(testing)
cluster <- kmeans(subset(training,select=-c(Species)),center=length(levels(iris$Species)))
training$cluster <- as.factor(cluster$cluster)
qplot(Petal.Width,Petal.Length,color=cluster,data=training)
table(cluster$cluster,training$Species)
modFit <- train(cluster ~ .,data=subset(training,select=-c(Species)),method="rpart")
table(predict(modFit,training),training$Species)
testPred <- predict(modFit,testing)
table(testPred,testing$Species)
