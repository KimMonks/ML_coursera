library("caret")
library("kernlab")
library(ggplot2)
library(Hmisc)
data(spam)
inTrain <- createDataPartition(y=spam$type,p=.75,list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
#Preprocess
hist(training$capitalAve)
preObj <- preProcess(training[,-58],method=c('center','scale'))
trainCapAveS <- predict(preObj,training[,-58])$capitalAve
preObj <- preProcess(training[,-58],method=c('BoxCox'))
trainCapAveS <- predict(preObj,training[,-58])$capitalAve
qqnorm(trainCapAveS)

dim(training)
dim(testing)
set.seed(12345)
modelFit <- train(type~.,data=training,method="glm",preProcess=c('center','scale'))
modelFit$finalModel
predictions<-predict(modelFit,newdata=testing)
confusionMatrix(predictions,testing$type)

#Impute
training$capAve <- training$capitalAve
selectNA <- rbinom(dim(training)[1],size=1,prob=.05)==1
training$capAve[selectNA] <- NA
preObj <- preProcess(training[,-58],method='knnImpute')
capAve <- predict(preObj,training[,-58])$capAve
capAveTruth<-training$capitalAve
capAveTruth <- (capAveTruth-mean(capAveTruth))/sd(capAveTruth)
quantile(capAve-capAveTruth)
quantile((capAve-capAveTruth)[selectNA])
quantile((capAve-capAveTruth)[!selectNA])

#Fold
folds <- createFolds(y=spam$type,k=10,list=TRUE,returnTrain=TRUE)
sapply(folds,length)
folds[[1]][1:10]
#Resample
folds <- createResample(y=spam$type,times=10,list=TRUE)
sapply(folds,length)
folds[[1]][1:10]
#Time slice
tme <- 1:1000
folds <- createTimeSlices(y=tme,initialWindow=20,horizon=5)
names(folds)
folds$train[[1]][1:21]
folds$test[[1]][1:6]

library(ISLR)
data(Wage)
summary(Wage)
inTrain <- createDataPartition(y=Wage$wage,p=.75,list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
featurePlot(x=training[,c('age','education','jobclass')],
            y=training$wage,
            plot="pairs")
qplot(age,wage,data=training,color=jobclass) + geom_smooth(method='lm',formular=y~x)

#Quanlity -> Indicator
cutWage <- cut2(training$wage,g=3)
table(cutWage)

p1 <- qplot(cutWage,age,data=training,fill=cutWage,geom=c('boxplot','jitter'))
p1

t1 <- table(cutWage,training$jobclass)
t1

qplot(wage,color=education,data=training,geom='density')

#Covariates
#Factor->Indicator
inTrain <- createDataPartition(y=Wage$wage,p=.75,list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
table(training$jobclass)
dummies <- dummyVars(wage~jobclass,data=training)
dummiesCol <- predict(dummies,training)
#Remove zero covariates
nsv <- nearZeroVar(training,saveMetrics=TRUE)
nsv
#Spline
library(splines)
bsBasis <- bs(training$age,df=3)
bsBasis
lml <- lm(wage~bsBasis,data=training)
plot(training$age,training$wage,pch=19,cex=0.5)
points(training$age,predict(lml,training),col='red',pch=19,cex=0.5)
predict(bsBasis,age=testing$age)

#PCA
inTrain <- createDataPartition(y=spam$type,p=.75,list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
M <- abs(cor(training[,-58]))
diag(M)<-0
which(M>0.8,arr.ind=T)
names(spam)[c(34,32)]*
plot(spam[,34],spam[,32])

X<-0.71*training$num415 + 0.71*training$num857
Y<-0.71*training$num415 - 0.71*training$num857
plot(X,Y)

smallSpam <- spam[,c(34,32)]
prComp <- prcomp(smallSpam)
prComp$rotation

typeColor <- ((spam$type=="spam")*1+1)
prComp <- prcomp(log10(spam[,-58]+1))
plot(prComp$x[,1],prComp$x[,2],col=typeColor,xlab="PC1",ylab="PC2")

preProc <- preProcess(log10(spam[,-58]+1),method="pca",pcaComp=2)
spamPC <- predict(preProc,log10(spam[,-58]+1))
plot(spamPC[,1],spamPC[,2],col=typeColor)
#Real
preProc <- preProcess(log10(training[,-58]+1),method="pca",pcaComp=10)
trainPC <- predict(preProc,log10(training[,-58]+1))
modelFit <- train(training$type~.,method='glm',data=trainPC)
testPC <- predict(preProc,log10(testing[,-58]+1))
confusionMatrix(testing$type,predict(modelFit,testPC))
