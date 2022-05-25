#===#===#===#===#===#===#===#===#===#===#===#===#===#
# An Introduction to Machine Learning               # 
#         for the Behavioural and Social Sciences   #
# Session #4: Resampling Methods                    #
# Gyeongcheol Cho and Heungsun Hwang                #
#===#===#===#===#===#===#===#===#===#===#===#===#===#

#install.packages("boot")  # for resampling methods 

# 1. Load libraries and data ----
library(boot)

# 0. Set the working directory
setwd('C:\\Users\\cheol\\Dropbox\\Teaching\\Workshops\\Introduction_to_Machine_Learning')
mydata=read.csv("auto_mpg.csv")
View(mydata)

# 2. Cross Validation ----
# 2-1. Validation Set Approach ----
ID = sample(392,392)
mydata.reordered = mydata[ID,]
View(mydata)
View(mydata.reordered)

mydata.training = mydata.reordered[1:196,] 
mydata.validation = mydata.reordered[197:392,]

plm.fit = lm(mpg ~ horsepower,data=mydata.training)
plm.fit = lm(mpg ~ poly(horsepower,2),data=mydata.training)
plm.fit = lm(mpg ~ poly(horsepower,3),data=mydata.training)

mean((mydata.validation$mpg - predict(plm.fit, mydata.validation))^2)
mean((mydata.validation$mpg - predict(plm.fit, mydata.validation))^2)
mean((mydata.validation$mpg - predict(plm.fit, mydata.validation))^2)

List.poly=1:10
list.mse.vsa=rep(0,10)
for (i in List.poly){
  plm.fit.vsa = lm(mpg ~ poly(horsepower,i),data=mydata.training)
  list.mse.vsa[i] = mean((mydata.validation$mpg - predict(plm.fit.vsa, mydata.validation))^2)
}
plot(List.poly,list.mse.vsa, type="b",main="Validaiton set approach",xlab="degree of polynomial",ylab="MSE",col="red")

# 2-2. LOOCV Approach ----
list.mse.loocv=rep(0,10)
for (i in List.poly){
  plm.fit.loocv = glm(mpg ~ poly(horsepower,i),data=mydata)
  list.mse.loocv[i] = cv.glm(mydata,plm.fit.loocv)$delta[1]
}
plot(List.poly,list.mse.loocv,type="b",main="LOOCV",xlab="degree of polynomial",ylab="MSE",col="red")

# 2-3. K-fold Approach ----
list.mse.loocv=rep(0,10)
for (i in List.poly){
  plm.fit.loocv = glm(mpg ~ poly(horsepower,i),data=mydata)
  list.mse.loocv[i] = cv.glm(mydata,plm.fit.loocv,K=10)$delta[1]
}
plot(List.poly,list.mse.loocv,type="b",main="K-fold approach",xlab="degree of polynomial",ylab="MSE",col="red")

plm.fit=lm(mpg ~ poly(horsepower,2),data=mydata)
summary(plm.fit)
plot(mpg ~ horsepower, data=mydata)
horsepower.sorted=sort(mydata$horsepower,index.return=TRUE)
lines(x = horsepower.sorted$x,
      y = plm.fit$fitted.values[horsepower.sorted$ix], 
      col="red")

# 2-4. K-fold Cross validation that can be used for general purpose  ----
# 2-4-1. specify a set of hyperparameter values considered and choose K for K-fold CV 
list.para=1:10
K=10
# 2-4-2. specify how to fit the model and to calculate validation error
genCVE = function(mydata,hp,loc.test){
  mydata.test=mydata[loc.test,]
  mydata.train=mydata[-loc.test,]
  rm(mydata)
  ####################### Users need to specify ############################
  model.fit = lm(mpg ~ poly(horsepower,hp),data=mydata.train) # for training 
  mean((mydata.test$mpg - predict(model.fit, mydata.test))^2) # for testing
  ##########################################################################
}
# 2-4-3. load the CVE function 
load("CVE.RData")
# 2-4-4. Run K-CV
Result=CVE(mydata,genCVE,list.para,K)

# 3. Bootstrapping methods ----
# 3-1. Train the model
model.fit = lm(mpg ~ horsepower+I(horsepower^2),data=mydata)
summary(model.fit)
# 3-2. Specify how to obtain parameter estimates of interest
genEst = function(mydata,loc.train){
  mydata.train=mydata[loc.train,]
  rm(mydata)
  ####################### Users need to specify ############################
  model.fit = lm(mpg ~ horsepower+I(horsepower^2),data=mydata.train) # for training 
  coef(model.fit)
  #summary(model.fit)$r.squared
  ##########################################################################
}
SE.boot=boot(mydata,genEst,1000)
boot.ci(SE.boot,type=c("basic"),index=1)
boot.ci(SE.boot,type=c("basic"),index=2)
boot.ci(SE.boot,type=c("basic"),index=3)
