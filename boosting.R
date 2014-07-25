library(ISLR)
data(Wage)
library(ggplot2)
library(caret)
Wage <- subset(Wage,select=-c(logwage))
inTrain <- createDataPartition(y=Wage$wage,p=.7,list=F)
training<-Wage[inTrain,];testing<-Wage[-inTrain]
modFit <- train(wage~.,model="gbm",data=training,verbose=F)
print(modFit)

