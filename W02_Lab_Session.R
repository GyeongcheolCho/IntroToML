#===#===#===#===#===#===#===#===#===#===#
# PSYC 493 & 747                        #
# Week #2: Regression                   #
# Heungsun Hwang and Gyeongcheol Cho    #
#===#===#===#===#===#===#===#===#===#===#

install.packages("psych")  # for calculating descriptive statistics 

# 0. Setting the working directory
  setwd('C:\\Users\\cheol\\Dropbox\\Mcgill\\Lecture\\Machine_Learning_Hwang\\W02_Regression')
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
  pred_y_test = predict(lm.fit.short, mydata_test)
  MSE = mean((mydata_test$anti1 - pred_y_test)^2)
  RMSE = sqrt(MSE)

# 3. MLR with dummy variables ----
  mydata = read.csv("GlastonburyDummy.csv") 
  View(mydata)
  mydata$music = factor(mydata$music, levels = c(4,1,2,3), labels = c('others','indiekid','metaller','crusty')) 
    # "others" will be used as a baseline in regression. 
  View(mydata)
  summary(mydata)
  
  lm.fit = lm(change ~ music, mydata)
  summary(lm.fit)

# 4. MLR with interaction terms ----
  mydata = read.csv("hospital.csv") 
  View(mydata)
  describe(mydata)
  lm.fit = lm(safety ~ exhaust*tenure+sex+age, mydata)
    # 'exhaust*tenure' can be replaced with 'exhaust+tenure+exhaust:tenure'
  summary(lm.fit)

  ## Draw the plot for interaction effect ----
  #install.packages("interactions") 
  library(interactions)
  interact_plot(lm.fit, pred = exhaust, modx = tenure)
  
# 5. Polynomial regression ----
  mydata = read.csv("curran_training.csv")
  View(mydata)
  describe(mydata)
  lm.fit = lm(anti1 ~ gender+cogstm+emotsup+I(emotsup^2), mydata)
  summary(lm.fit)

# 6. KNN regression ----
  ## Run a model ----
  #install.packages("caret")
  library(caret)
  mydata = read.csv("curran_training.csv")
  mydata_test = read.csv("curran_test.csv") 
  knnmodel = knnreg(anti1 ~ gender+cogstm+emotsup, mydata, k = 5)
  
  ## Calculate R2 ----
  pred_y_knn = predict(knnmodel, mydata)
  R2_knn = 1 - sum((mydata$anti1 - pred_y_knn)^2)/sum((mydata$anti1 - mean(mydata$anti1))^2)

  lm.fit1 = lm(anti1 ~ gender+cogstm+emotsup, mydata) # multiple linear regression
  R2_lm1 = 1 - sum(lm.fit1$residuals^2)/sum((mydata$anti1 - mean(mydata$anti1))^2)
  
  cat("\n R2_knn: ", R2_knn, "\n R2_lm1: ", R2_lm1)
  
# 7. Model Comparison (MSE) ----
  ## Run competing models ----
  lm.fit1 = lm(anti1 ~ gender+cogstm+emotsup, mydata) # multiple linear regression
  lm.fit2 = lm(anti1 ~ gender+cogstm*emotsup, mydata) # multiple linear regression w/ interaction
  lm.fit3 = lm(anti1 ~ gender+cogstm+emotsup+I(cogstm^2), mydata) # Polynomial regression w/ quadratic term 
  lm.fit4 = lm(anti1 ~ gender+cogstm*emotsup+I(cogstm^2), mydata) # Polynomial regression w/ interaction & quadratic term 

  ## Calculate MSE values ---- 
  pred_y_knn = predict(knnmodel, mydata_test)
  pred_y_lm1 = predict(lm.fit1, mydata_test)
  pred_y_lm2 = predict(lm.fit2, mydata_test)
  pred_y_lm3 = predict(lm.fit3, mydata_test)
  pred_y_lm4 = predict(lm.fit4, mydata_test)

  MSE_knn = mean((mydata_test$anti1 - pred_y_knn)^2)
  MSE_lm1 = mean((mydata_test$anti1 - pred_y_lm1)^2)
  MSE_lm2 = mean((mydata_test$anti1 - pred_y_lm2)^2)
  MSE_lm3 = mean((mydata_test$anti1 - pred_y_lm3)^2)
  MSE_lm4 = mean((mydata_test$anti1 - pred_y_lm4)^2)
  
  cat("\n MSE_knn: ", MSE_knn, "\n MSE_lm1: ", MSE_lm1,
      "\n MSE_lm2: ", MSE_lm2, "\n MSE_lm3: ", MSE_lm3,
      "\n MSE_lm4: ", MSE_lm4)

  cat("\n\n\n\n\n MSE_knn: ", MSE_knn)
  