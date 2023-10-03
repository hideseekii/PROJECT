library(rpart) #install.packages("rpart")
library(rpart.plot) #install.packages("rpart.plot")
library("randomForest") #install.packages("randomForest")
library("performanceEstimation") 

#清空暫存資料
rm(list=ls())

########################### 先預測錯誤出現 #####################################

Failure <- function(train_data,test_data){
  
  
  set.seed(123)
  
  #RF_Failure 錯誤出現的隨機森林樹
  #result_Failure 錯誤測結果
  
  RF_Failure = randomForest(Target~Air.temperature..K.+Process.temperature..K.+Rotational.speed..rpm.+Torque..Nm.+Tool.wear..min.+Type
                            , data = train_data, importance=T, proximity=T, do.trace=100,ntree=500)
  
  
  result_Failure = predict(RF_Failure , newdata = test_data)
  CM = table(result_Failure,test_data$Target)
  
  ######################### 資料合併看預測結果與實際結果 #######################
  
  #print("混淆矩陣")
  
  #print(CM)

  #print("預測是否發生Failure準確率")
  
  #print(sum(diag(CM))/sum(CM))
  
  
  #結果寫成新的test data
  
  temp_data = test_data
  
  for(i in 1:nrow(temp_data)){
    
    temp_data$Failure.Type = "No Failure"
    
  }
  
  temp_data$Target = result_Failure

  return (temp_data)
}


########################### 預測錯誤種類 #######################################
FailureType = function(train_data,test_data,temp_data){
  
  Failuretype_train = split(train_data,train_data$Failure.Type != "No Failure")
  
  Failuretype_train = as.data.frame(Failuretype_train[[2]])
  
  Failuretype_train = droplevels(Failuretype_train)

  set.seed(123)
  
  #RF_Failuretype是預測種類的隨機森林
  
  RF_Failuretype = randomForest(Failure.Type~Air.temperature..K.+Process.temperature..K.+Rotational.speed..rpm.+Torque..Nm.+Tool.wear..min.+Type
                                , data = Failuretype_train, importance=T, proximity=T, do.trace=100,ntree=500)
  
  for (i in 1:nrow(temp_data)) {
    
    if (temp_data$Target[i] == '1') {
      
      # 使用 newdata 參數傳遞資料框
      prediction <- predict(RF_Failuretype, newdata = temp_data[i, ])
      
      # 印出預測結果
      temp_data$Failure.Type[i] = as.character(prediction)
    }
    
  }
  
  result = temp_data$Failure.Type
  CM=table(result,test_data$Failure.Type)
  
  #print(result)
  
  #print("混淆矩陣")
  
  #print(CM)
  
  
  #print("預測FailureType的準確率")
  
  #print(sum(diag(CM))/sum(CM))
  
  return (sum(diag(CM))/sum(CM))
}

RF_Prediction = function(train_data,test_data){
  
  train_data$Target = as.factor(train_data$Target)
  
  print("Failure預測")
  
  temp_data = Failure(train_data,test_data)
  
  print("Type預測")
  
  result = FailureType(train_data,test_data,temp_data)
  
  print("準確率:")
  print(result)
  
  
  return (result)
}






############################# 主要程式 #########################################
#讀取檔案資料
#讀取檔案，header = TRUE表示第一筆資料為標題，stringsAsFactors = FALSE表示不自動設定分類
dp=read.csv("predictive_maintenance.csv", header=TRUE, stringsAsFactors = TRUE, fileEncoding = 'utf-8')
#建立引數(就可以直接使用欄位名稱)
attach(dp)


########################### 交叉驗證資料 #######################################
library(rsample)
library(caret)

data <- dp  # 載入你的資料


#### 資料前處理 ####
## 1. Target 和 Failure.Type 是否一致

# is "1" but "No Failure"
drop1 = which(data[which(data$Target == 1),]$Failure.Type == "No Failure")
data = data[-drop1,]

# is "0" but "Failure"
drop2 = which(data[which(data$Target == 0),]$Failure.Type != "No Failure")
data = data[-drop2,]


# 創建5折交叉驗證數據
cv_splits <- vfold_cv(data, v = 5)

set.seed(13)

folds <- vfold_cv(data, v = 5)

result_list = list()

############################# 隨機森林樹 #######################################
for (i in 1:5){
  
  train_data = analysis(folds$splits[[i]])
  
  test_data  = assessment(folds$splits[[i]])
  
  result = RF_Prediction(train_data,test_data)
  
  result_list = c(result_list , result)
}


print("K-Fold 交叉驗證隨機森林樹結果")
print(mean(unlist(result_list)))



############################# KNN演算法 ########################################


