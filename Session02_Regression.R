#===#===#===#===#===#===#===#===#===#===#===#===#===#
# An Introduction to Machine Learning               # 
#         for the Behavioural and Social Sciences   #
# Session #2: Linear Regression                     #
# Heungsun Hwang and Gyeongcheol Cho                #
#===#===#===#===#===#===#===#===#===#===#===#===#===#

#install.packages("psych")  # for calculating descriptive statistics 
#install.packages("caret")  # for KNN regression


# 0. Set the working directory
  setwd('C:\\Users\\cheol\\Dropbox\\Teaching\\Workshops\\Introduction_to_Machine_Learning')
    # Note that "/" or "\\" should be used for path separator, not "\"  

# 1. Simple linear regression (SLR) ----
  mydata = read.csv("wineheartattack.csv") 
  View(mydata)
  summary(mydata)
  
  describe(mydata)
    library(psych)
  describe(mydata)
  
  lm.fit = lm(heartattack ~ wine, mydata) 
  summary(lm.fit)
  plot(mydata$wine, mydata$heartattack)
  abline(lm.fit, lwd = 3, col = "red")
  
    lm.fit2 = lm(heartattack ~ wine + I(wine^2), mydata)
    summary(lm.fit)
    pred_y_tr = predict(lm.fit2, mydata)
    ix = sort(mydata$wine, index.return=T)$ix
    lines(mydata$wine[ix], pred_y_tr[ix], lwd = 3, col = "red")
  
# 2. Multiple linear regression (MLR) ----
  ## Run a model ----
  mydata = read.csv("curran_training.csv") 
  View(mydata)
  colnames(mydata)
  colnames(mydata)[1] = "ID"
  colnames(mydata)

  summary(mydata)
  describe(mydata)
  
  lm.fit.long = lm(anti1 ~ gender + cogstm + emotsup, mydata)
  summary(lm.fit.long)
  
  lm.fit.short = lm(anti1 ~ gender+emotsup, mydata)
  summary(lm.fit.short)
  
  ## Calculate predicted values/MSE/RMSE ----
  mydata_test = read.csv("curran_test.csv") 
  pred_y_test = predict(lm.fit.long, mydata_test)

  RSE = sqrt(sum((mydata_test$anti1 - pred_y_test)^2)/(nrow(mydata_test)-3-1))
  MSE = mean((mydata_test$anti1 - pred_y_test)^2)
  RMSE = sqrt(MSE)

  mydata_test$anti1
  pred_y_test
  RSE
  MSE
  RMSE
  
# 3. Polynomial regression ----
  lm.fit = lm(anti1 ~ gender+cogstm+emotsup+I(emotsup^2), mydata)
  summary(lm.fit)
    
# 4. KNN regression ----
  ## Run a model ----
  library(caret)
  knnmodel = knnreg(anti1 ~ gender+cogstm+emotsup, mydata, k = 5)
  
  ## Calculate R2 ----
  pred_y_knn = predict(knnmodel, mydata)
  TSS = sum((mydata$anti1 - mean(mydata$anti1))^2)
  RSS = sum((mydata$anti1 - pred_y_knn)^2)
  R2_knn = 1 - RSS/TSS

  lm.fit1 = lm(anti1 ~ gender+cogstm+emotsup, mydata) # multiple linear regression
  R2_lm1 = 1 - sum(lm.fit1$residuals^2)/TSS

  lm.fit2 = lm(anti1 ~ gender+cogstm+emotsup+I(emotsup^2), mydata) # Polynomial regression w/ quadratic term 
  R2_lm2 = 1 - sum(lm.fit2$residuals^2)/TSS
  
  cat("\n R2_knn: ", R2_knn, "\n R2_lm1: ", R2_lm1, R2_lm2)
  
# 5. Model Comparison (MSE) ----
  ## Calculate MSE values ---- 
  pred_y_knn = predict(knnmodel, mydata_test)
  pred_y_lm1 = predict(lm.fit1, mydata_test)
  pred_y_lm2 = predict(lm.fit2, mydata_test)

  MSE_knn = mean((mydata_test$anti1 - pred_y_knn)^2)
  MSE_lm1 = mean((mydata_test$anti1 - pred_y_lm1)^2)
  MSE_lm2 = mean((mydata_test$anti1 - pred_y_lm2)^2)

  cat("\n MSE_knn: ", MSE_knn,
      "\n MSE_lm1: ", MSE_lm1,
      "\n MSE_lm2: ", MSE_lm2)