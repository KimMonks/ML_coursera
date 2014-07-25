data(iris)
library(caret)
library(ggplot2)
inTrain <- createDataPartition(y=iris$Species,p=.7,list=F)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
modFit <- train(Species~.,data=training,method="rf",prox=T)
getTree(modFit$finalModel,k=500)

# Class "centers"
irisP  <- classCenter(training[,c(3,4)],training$Species,modFit$finalModel$proximity)
irisP <- as.data.frame(irisP); irisP$Species  <- row.names(irisP)
p <- qplot(Petal.Width,Petal.Length,col=Species,data=training)
p + geom_point(aes(x=Petal.Width,y=Petal.Length,col=Species),data=irisP,size=5,shape=4)
# Predict new values
pred <- predict(modFit,testing); testing$predRight <- pred == testing$Species
table(pred,testing$Species)
qplot(Petal.Width,Petal.Length,col=predRight,data=testing)

rfcv()