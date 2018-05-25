# ---
# title: "ad.tracking_modeling"
# author: "kimi"
# date: "2018?뀈 5?썡 18?씪"
# output: html_document
# ---

# Load libraries ----------------------------------------------------------------------------

library(data.table)
library(sqldf)
library(ggplot2)
library(gridExtra)
library(caret)
library(dplyr)
library(lubridate)  # time 
library(xgboost)


# Load data ----------------------------------------------------------------------------------

train_path <- "../data/ad.tracking/train_10m.csv"
#train_sample_path <- "../data/ad.tracking/train_sample.csv"
test_path  <- "../data/ad.tracking/test.csv"

#subm_path <- "../data/ad.tracking/sample_submission.csv"
#train_sample_path <- "../data/ad.tracking/train_sample.csv"

train <- fread(train_path, showProgress = T, stringsAsFactors = F, na.string=c("", NA))
test <- fread(test_path, showProgress = T, stringsAsFactors = F)

# subm <- fread(subm_path, showProgress = F, stringsAsFactors = F)
# df <- fread(train_sample_path, showProgress = F, stringsAsFactors = F, na.string=c("", NA))

#set.seed(1234)
#train <- train[sample(.N, 1e6), ]


# train data pre-processing ------------------------------------------------------------------

######################################## 
print("01 Train data pre-processing start ----------------")
######################################## 

tr2 <- train %>% select(-c(attributed_time)) %>% 
  mutate(day = day(click_time), hour = hour(click_time)) %>%
  select(-c(click_time)) %>%
  add_count(app) %>% rename("app_cnt" = n) %>%
  add_count(channel) %>% rename("chl_cnt" = n) %>%
  add_count(app, channel) %>% rename("app_chl" = n) %>%
  add_count(app, channel, ip) %>% rename("app_chl_ip" = n) %>%
  add_count(app, channel, os) %>% rename("app_chl_os" = n) %>% 
  add_count(app, channel, device) %>% rename("app_chl_device" = n) %>%
  add_count(app, channel, day) %>% rename("app_chl_day" = n) %>% 
  add_count(app, channel, hour) %>% rename("app_chl_hour" = n) 

head(tr2)


######################################## 
print("01 Train data pre-processing end ----------------")
print(head(tr2, 10))
######################################## 


## Train/test data partition -----------------------------------------------------------------

######################################## 
print("02 Data partition start ----------------")
######################################## 

# Transforming the dependent variable to a factor
tr2$is_attributed <- as.factor(tr2$is_attributed)

# Partitioning the data into train/valid data
set.seed(1234)
tri <- createDataPartition(tr2$is_attributed, p=0.7, list=F)

train <- tr2[tri, ]
valid <- tr2[-tri, ]

# Setting levels for both train/valid data
levels(train$is_attributed) <- make.names(levels(factor(train$is_attributed)))
levels(valid$is_attributed) <- make.names(levels(factor(valid$is_attributed)))


######################################## 
print("02 Data partition end ----------------")
print(dim(train))
print(dim(valid))
######################################## 


## Modeling: Random Forest ------------------------------------------------------------

######################################## 
print("03 Modeling - RF start ----------------")
######################################## 

# Setting up train controls
repeats <- 3
numbers <- 10
tunel <- 10

set.seed(1234)
ctrl = trainControl(method="repeatedcv",
                    number=numbers,
                    repeats = repeats,
                    classProbs=TRUE,
                    summaryFunction = twoClassSummary)

# model 1
m_rf1 <- train(is_attributed~., data=train, 
               method = "rf",
               metric="ROC",
               trControl=ctrl)


install.packages("randomForest")
library(randomForest)

# model 2
set.seed(1234)
m_rf2 <- randomForest(is_attributed~., data=train, proximity=FALSE, importance=TRUE, ntree=500, mtry=3)

m_rf2


######################################## 
print("03 Modeling - RF end ----------------")
print(summary(m_rf1))
print(summary(m_rf2))
######################################## 


# Model Evaludation ----------------------------------------------------------------------------

######################################## 
print("04. Model prediction & evaluation start ----------------")
######################################## 

p.valid1 <- predict(m_rf1, valid, type = "prob")

#install.packages("ROCR")
library(ROCR)

pr_rf1 <- prediction(p.valid1[, 2], valid$is_attributed)
prf_rf1 <- performance(pr_rf1, "tpr", "fpr")
auc_rf1 <- performance(pr_rf1, "auc") 
(auc_rf1 <- auc_rf1@y.values[[1]]) 

#plot(prf_rf1, main="ROC curve with random forest model")
#text(0.5, 0.5, paste0("auc:",round(auc_rf1, 3)), cex=1.5, col="red")


p.valid2 <- predict(m_rf2, valid, type = "prob")

pr_rf2 <- prediction(p.valid2[, 2], valid$is_attributed)
prf_rf2 <- performance(pr_rf2, "tpr", "fpr")
auc_rf2 <- performance(pr_rf2, "auc") 
(auc_rf2 <- auc_rf2@y.values[[1]]) 


######################################## 
print("04. Model evaluation end ----------------")
print(paste0("auc 1:", round(auc_rf1, 3)))
print(paste0("auc 2:", round(auc_rf2, 3)))
    
######################################## 


## Subminssion prep --------------------------------------------------------------------------

######################################## 
print("05 Subminssion prep start ----------------")
######################################## 

sub1 <- data.table(click_id = test$click_id, is_attributed = NA)
sub2 <- data.table(click_id = test$click_id, is_attributed = NA)
test$click_id <- NULL

head(sub)

######################################## 
print("05 Subminssion prep end ----------------")
######################################## 



## test data preprocessing -------------------------------------------------------------------

######################################## 
print("06 Test data preprocessing start ----------------")
######################################## 

test <- test %>% mutate(day = day(click_time), hour = hour(click_time)) %>%
  select(-c(click_time)) %>%
  add_count(app) %>% rename("app_cnt" = n) %>%
  add_count(channel) %>% rename("chl_cnt" = n) %>%
  add_count(app, channel) %>% rename("app_chl" = n) %>%
  add_count(app, channel, ip) %>% rename("app_chl_ip" = n) %>%
  add_count(app, channel, os) %>% rename("app_chl_os" = n) %>% 
  add_count(app, channel, device) %>% rename("app_chl_device" = n) %>%
  add_count(app, channel, day) %>% rename("app_chl_day" = n) %>% 
  add_count(app, channel, hour) %>% rename("app_chl_hour" = n) 

head(test)

######################################## 
print("06 Test data preprocessing end ----------------")
print(head(test2, 10))
######################################## 



## Prediction --------------------------------------------------------------------------------

######################################## 
print("07 Prediction start ----------------")
######################################## 

p.test1 <- predict(m_rf1, test2, type = "prob")
sub1$is_attributed = p.test1

p.test2 <- predict(m_rf2, test2, type = "prob")
sub2$is_attributed = p.test2

######################################## 
print("07 Prediction end ----------------")
######################################## 



## Create submission file --------------------------------------------------------------------

######################################## 
print("08 fwrite start ----------------")
######################################## 

fwrite(sub1, "sub_rf1_R_10m.csv")
fwrite(sub2, "sub_rf2_R_10m.csv")

######################################## 
print("08 fwrite end ----------------")
######################################## 


