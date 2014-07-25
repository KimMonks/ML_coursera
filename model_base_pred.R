library(ggplot2)
library(caret)
data(iris)
names(iris)
table(iris$Species)
inTrain <- createDataPartition(y=iris$Species,p=.7,list=F)
training<-iris[inTrain,]
testing<-iris[-inTrain,]
dim(training); dim(testing)

modlna <- train(Species~.,data=training,method="lda")
modnb  <- train(Species~.,data=training,method="nb")
plda  <- predict(modlna,testing)
pnb  <- predict(modnb,testing)
equalPred  <- (plda==pnb)
qplot(Petal.Width,Sepal.Width,color=equalPred,data=testing)
