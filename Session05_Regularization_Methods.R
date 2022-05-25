#===#===#===#===#===#===#===#===#===#===#===#===#===#
# An Introduction to Machine Learning               # 
#         for the Behavioural and Social Sciences   #
# Session #5: Regularization Methods                #
# Heungsun Hwang and Gyeongcheol Cho                #
#===#===#===#===#===#===#===#===#===#===#===#===#===#

install.packages('glmnet') # for Ridge/Lasso regression
# 1. Load libraries and data ---- 
library(glmnet) 
setwd('C:\\Users\\cheol\\Dropbox\\Teaching\\Workshops\\Introduction_to_Machine_Learning')

mydata = read.csv('Hitters_training.csv')
View(mydata)
mydata$League =as.factor(mydata$League) 
mydata$Division = as.factor(mydata$Division)
mydata$NewLeague =as.factor(mydata$NewLeague)
mydata.X=data.matrix(mydata[,colnames(mydata)!="Salary"])

mydata_test = read.csv('Hitters_test.csv')
mydata_test$League =as.factor(mydata_test$League) 
mydata_test$Division = as.factor(mydata_test$Division)
mydata_test$NewLeague =as.factor(mydata_test$NewLeague)
mydata_test.X=data.matrix(mydata_test[,colnames(mydata_test)!="Salary"])

# 2. Regularization methods ----
##2-1. Given lambda ----
###2-1-1. Ridge regression ----
  lambdas = c(10^9,10^6,10^3,10^0) # make a list of lambdas
  ridge.fit= glmnet(mydata.X, mydata$Salary,
                    alpha = 0, lambda = lambdas)
  ridge.coef=coef(ridge.fit)
  colnames(ridge.coef)=c("lam=10^9","lam=10^6","lam=10^3","lam=10^0")
  ridge.coef
  
## 2-1-2. Lasso regression given lambda ----
  lambdas = c(10^1,10^0,10^-1,10^-2)
  lasso.fit= glmnet(mydata.X, mydata$Salary,
                    alpha = 1, lambda = lambdas)
  lasso.coef=coef(lasso.fit)
  colnames(lasso.coef)=c("lam=10^1","lam=10^0","lam=10^-1","lam=10^-2")
  lasso.coef
  
## 2-2. While searching for the best lambda ----
### 2-2-1. Ridge regression ----
  lambdas = 10^seq(from = 3,to = -2, length.out = 100)
  ridge.cv = cv.glmnet(mydata.X, mydata$Salary,
                       alpha = 0, lambda = lambdas,
                       nfold = 5)
  plot(lambdas[100:1],ridge.cv$cvm[100:1],xlab="lambda",ylab="MSE")
  ridge.lam=ridge.cv$lambda.min
  ridge.fit= glmnet(mydata.X, mydata$Salary,
                    alpha = 0, lambda = ridge.lam)
  
### 2-2-2. Lasso regression ----
  lambdas = 10^seq(from = 2,to = -3, length.out = 100)
  lasso.cv = cv.glmnet(mydata.X, mydata$Salary,
                       alpha = 1, lambda = lambdas,
                       nfold = 5)
  plot(lambdas[100:1],lasso.cv$cvm[100:1],xlab="lambda",ylab="MSE")
  lasso.lam=lasso.cv$lambda.min
  lasso.fit= glmnet(mydata.X, mydata$Salary,
                    alpha = 1, lambda = lasso.lam)
  
## 2-3. Performance Evaluation ----
  # Ridge
  ridge.pred.y.tt = predict(ridge.fit,newx = mydata_test.X)
  ridge.MSE = mean((ridge.pred.y.tt - mydata_test$Salary)^2)
  ridge.MSE
  # Lasso
  lasso.pred.y.tt = predict(lasso.fit,newx = mydata_test.X)
  lasso.MSE = mean((lasso.pred.y.tt - mydata_test$Salary)^2)
  lasso.MSE
  # Linear regression
  lm.fit = lm(Salary ~ ., mydata)
  lm.pred.y.tt = predict(lm.fit,mydata_test)
  lm.MSE = mean((lm.pred.y.tt - mydata_test$Salary)^2)
  
  # Results 
  result = cbind(coef(ridge.fit),coef(lasso.fit),coef(lm.fit))
  colnames(result)=c('Ridge','Lasso','No Reg')
  result
  cat("\n MSE_Ridge: ", ridge.MSE,
      "\n MSE_Lasso: ", lasso.MSE,
      "\n MSE_Basic: ", lm.MSE)
  