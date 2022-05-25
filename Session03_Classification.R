#===#===#===#===#===#===#===#===#===#===#===#===#===#
# An Introduction to Machine Learning               # 
#         for the Behavioural and Social Sciences   #
# Session #3: Classification Methods                #
# Heungsun Hwang and Gyeongcheol Cho                #
#===#===#===#===#===#===#===#===#===#===#===#===#===#

#install.packages('caret') # for making confusion Matrix
#install.packages('pROC')  # for plotting ROC curve
#install.packages('MASS')  # for conducting LDA/QDA
#install.packages('e1071')  # for Naive Bayes
#install.packages('class')  # for conducting KNN  

# 1. Load libraries and data
  library(caret) 
  library(pROC)
  library(MASS)
  library(e1071)
  library(class)
  setwd('C:\\Users\\cheol\\Dropbox\\Teaching\\Workshops\\Introduction_to_Machine_Learning')
  
  mydata = read.csv("BBB_training.csv") 
  head(mydata)
  colnames(mydata)[1]="id"
  mydata$buyer = as.factor(mydata$buyer)
#  mydata$gender = as.factor(mydata$gender)
  head(mydata)
    
  mydata_test = read.csv("BBB_test.csv") 
  head(mydata_test)
  colnames(mydata_test)[1]="id"
  mydata_test$buyer = as.factor(mydata_test$buyer)
#  mydata_test$gender = as.factor(mydata_test$gender)
  head(mydata_test)

# 2. Logistic Regression ----
  ## 2.1 Run a model ----
  log.fit = glm(buyer ~ gender + last + book + art + child
                         + youth + cook + do_it + reference + geog,
                         data = mydata , family = binomial)
  summary(log.fit)
  Exp_b_CI=exp(cbind(coef(log.fit), confint(log.fit))) # obtain estimates of exp(B) and their 95%CI 
  Exp_b_CI
  
  ## 2.2 Additional Analysis ----
  ### on training sample ----
  log.prob.y.tr = predict(log.fit, type = "response") # calculate individual probabilities
    # type = "response" tells R to calculate P(Y = 1|X) for each observation
  log.pred.y.tr = ifelse(log.prob.y.tr>.5,1,0) # calculate predicted values using .5 cutoff
  log.pred.y.tr = as.factor(log.pred.y.tr)
  confusionMatrix(data = log.pred.y.tr, reference = mydata$buyer, positive = "1") # obtain confusion matrix
  log.roc.tr = roc(mydata$buyer,log.prob.y.tr,auc=TRUE) # plot the ROC curve with AUC 
  plot(log.roc.tr,print.auc=TRUE, legacy.axes=TRUE,
        ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
        auc.polygon=TRUE,grid=TRUE)
    # legacy.axes = TRUE if the x-axis must be plotted as increasing FPR(1-specificity)
    # auc.polygon = TRUE if you want to color the area under the ROC curve. 
  
  ### on test sample ----
  log.prob.y.tt = predict(log.fit, mydata_test, type = "response")
  log.pred.y.tt = ifelse(log.prob.y.tt>.5,1,0)
  log.pred.y.tt = as.factor(log.pred.y.tt)
  confusionMatrix(data = log.pred.y.tt, reference = mydata_test$buyer, positive = "1")
  log.roc.tt = roc(mydata_test$buyer,log.prob.y.tt,auc=TRUE) # plot the ROC curve with AUC 
  plot(log.roc.tt,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE)

# 3. LDA ----
  ## 3.1 Run a model ----
  lda.fit <- lda(buyer ~ gender + last + book + art + child
                  + youth + cook + do_it + reference + geog,
                  data = mydata )

  ## 3.2 Additional Analysis ----
  ### on training sample ----
  lda.list.y.tr = predict(lda.fit)
  lda.prob.y.tr = lda.list.y.tr$posterior[,2]
  lda.pred.y.tr = lda.list.y.tr$class
  
  confusionMatrix(data = lda.pred.y.tr, reference = mydata$buyer, positive = "1") # obtain confusion matrix
  lda.roc.tr = roc(mydata$buyer,lda.prob.y.tr,auc=TRUE) # plot the ROC curve with AUC 
  plot(lda.roc.tr,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE)

  ### on test sample ----
  lda.list.y.tt = predict(lda.fit,mydata_test)
  lda.prob.y.tt = lda.list.y.tt$posterior[,2]
  lda.pred.y.tt = lda.list.y.tt$class
  
  confusionMatrix(data = lda.pred.y.tt, reference = mydata_test$buyer, positive = "1")
  lda.roc.tt = roc(mydata_test$buyer,lda.prob.y.tt,auc=TRUE) # plot the ROC curve with AUC 
  plot(lda.roc.tt,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE)
  
# 4. QDA ----
  ## 4.1 Run a model ----
  qda.fit <- qda(buyer ~ gender + last + book + art + child
                 + youth + cook + do_it + reference + geog,
                 data = mydata )
  ## 4.2 Additional Analysis ----
  ### on training sample ----
  qda.list.y.tr = predict(qda.fit)
  qda.prob.y.tr = qda.list.y.tr$posterior[,2]
  qda.pred.y.tr = qda.list.y.tr$class
  
  confusionMatrix(data = qda.pred.y.tr, reference = mydata$buyer, positive = "1") # obtain confusion matrix
  qda.roc.tr = roc(mydata$buyer,qda.prob.y.tr,auc=TRUE) # plot the ROC curve with AUC 
  plot(qda.roc.tr,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE)
  
  ### on test sample ----
  qda.list.y.tt = predict(qda.fit,mydata_test)
  qda.prob.y.tt = qda.list.y.tt$posterior[,2]
  qda.pred.y.tt = qda.list.y.tt$class
  
  confusionMatrix(data = qda.pred.y.tt, reference = mydata_test$buyer, positive = "1")
  qda.roc.tt = roc(mydata_test$buyer,qda.prob.y.tt,auc=TRUE) # plot the ROC curve with AUC 
  plot(qda.roc.tt,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE)

# 5. Naive Bayes ----
  ## 5.1 Run a model ----
  nb.fit <- naiveBayes(buyer ~ gender + last + book + art + child
                       + youth + cook + do_it + reference + geog,
                       data = mydata )
  
  ## 5.2 Additional Analysis ----
  ### on training sample ----
  nb.prob.y.tr = predict(nb.fit,mydata,type = "raw")[,2]
  nb.pred.y.tr = predict(nb.fit,mydata,type = "class")
  
  confusionMatrix(data = nb.pred.y.tr, reference = mydata$buyer, positive = "1") # obtain confusion matrix
  nb.roc.tr = roc(mydata$buyer,nb.prob.y.tr,auc=TRUE) # plot the ROC curve with AUC 
  plot(nb.roc.tr,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE)
  
  ### on test sample ----
  nb.prob.y.tt = predict(nb.fit,mydata_test,type = "raw")[,2]
  nb.pred.y.tt = predict(nb.fit,mydata_test,type = "class")
  
  confusionMatrix(data = nb.pred.y.tt, reference = mydata_test$buyer, positive = "1")
  nb.roc.tt = roc(mydata_test$buyer,nb.prob.y.tt,auc=TRUE) # plot the ROC curve with AUC 
  plot(nb.roc.tt,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE,grid=TRUE)
  
# 6. KNN ----
  ## 6.1 Preprocess data ----
  mydata.X = scale(mydata[,c(2:11)])
  mydata.Y = mydata[,12]
  mydata_test.X = scale(mydata_test[,c(2:11)])
  ## 6.2 Run KNN
  K=200
  knn.pred.y.tt=knn(mydata.X, mydata_test.X, mydata.Y, K,prob=TRUE)
  knn.prob.y.tt=attr(knn.pred.y.tt,"prob")
  
  confusionMatrix(data = knn.pred.y.tt, reference = mydata_test$buyer, positive = "1")
  knn.roc.tt = roc(mydata_test$buyer,knn.prob.y.tt,auc=TRUE) # plot the ROC curve with AUC 
  plot(knn.roc.tt,print.auc=TRUE,legacy.axes=TRUE,
       ylab = "True Positive Rate",xlab = "False Positive Rate",main = "ROC",
       auc.polygon=TRUE)
  
# 7. Model Comparison (AUC) ----
  plot(log.roc.tt,legacy.axes=TRUE,col="black",lwd=4,
       ylab = "True Positive Rate",xlab = "False Positive Rate",
       main = "ROC",grid=TRUE)
  plot(lda.roc.tt,legacy.axes=TRUE,col="red",lwd=4,add=TRUE)
  plot(qda.roc.tt,legacy.axes=TRUE,col="green",lwd=4,add=TRUE)
  plot(nb.roc.tt,legacy.axes=TRUE,col="purple",lwd=4,add=TRUE)
  plot(knn.roc.tt,legacy.axes=TRUE,col="blue",lwd=4,add=TRUE)
  legend("bottomright",legend=c("LR","LDA","QDA","NB","KNN"),col=c("black","red","green","purple","blue"),lwd=4)
  
  cat("\n AUC_LR:  ", log.roc.tt$auc,
      "\n AUC_LDA: ",lda.roc.tt$auc,
      "\n AUC_QDA: ",qda.roc.tt$auc,
      "\n AUC_NB:  ", nb.roc.tt$auc,
      "\n AUC_KNN: ",knn.roc.tt$auc)
  