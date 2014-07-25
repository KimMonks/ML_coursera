library(ggplot2)
library(caret)
library(ISLR)
data(Wage)
inBuild <- createDataPartition(y=Wage$wage,p=.7,list=F)
validation <- Wage[inBuild,]
buildData <- Wage[-inBuild,]
inTrain <- createDataPartition(y=buildData$wage,p=.7,list=F)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
dim(training)
dim(testing)
dim(validation)
mod1 <- train(wage~.,method="glm",data=training)
mod2 <- train(wage~.,method="rf",data=training,trControl=trainControl(method="cv"),number=3)
pred1 <- predict(mod1,validation)
pred2 <- predict(mod2,validation)
qplot(pred1,pred2,color=validation$wage)

predDF <- data.frame(pred1,pred2,wage=validation$wage)
combModFit <- train(wage~.,method="gam",data=predDF)
combPred <- predict(combModFit,validation)
qplot(x=combPred,y=validation$wage) + geom_abline(slope=1,perception=0)
