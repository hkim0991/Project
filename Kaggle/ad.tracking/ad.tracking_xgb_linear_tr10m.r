# ---
# title: "ad.tracking_modeling"
# author: "kimi"
# date: "2018???? 5???? 18????"
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

#train <- fread('C:\\Users\\Kim\\Documents\\R\\Home_Review\\R_review\\ad_tracking\\train_10m.csv', showProgress = F)
#test <- fread('C:\\Users\\Kim\\Documents\\R\\Home_Review\\R_review\\ad_tracking\\test.csv', showProgress = F)

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
invisible(gc())


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

invisible(gc())


######################################## 
print("02 Data partition end ----------------")
print(dim(train))
print(dim(valid))
######################################## 


## Modeling: Random Forest ------------------------------------------------------------

######################################## 
print("03 Modeling - XGboost start ----------------")
######################################## 

# Setting up train controls
repeats <- 3
numbers <- 10
tunel <- 10



# model 2
set.seed(1234)
ctrl2 = trainControl(method="repeatedcv",
                    number=numbers,
                    repeats = repeats,
                    classProbs=TRUE,
                    summaryFunction = twoClassSummary)

grid2 = expand.grid(nrounds = 100, 
                    eta = 0.2,
                    lambda = 1,
                    alpha = 0)

m_xgb2 <- train(is_attributed~., data=train, 
                method = "xgbLinear",
                metric="ROC",
                trControl=ctrl2,
                tuneLength=tunel,
                nthread = 4,
                tuneGrid = grid2)
               

#plot(m_dt1$finalModel, main="Classification Tree")
#text(m_dt1$finalModel)

invisible(gc())


######################################## 
print("03 Modeling - XGboost end ----------------")
print(m_xgb2)

######################################## 


# Model Evaludation ----------------------------------------------------------------------------

######################################## 
print("04. Model prediction & evaluation start ----------------")
######################################## 

p.valid2 <- predict(m_xgb2, valid, type = "prob")

#install.packages("ROCR")
library(ROCR)


# model 2
pr_xgb2 <- prediction(p.valid2[, 2], valid$is_attributed)
prf_xgb2 <- performance(pr_xgb2, "tpr", "fpr")
auc_xgb2 <- performance(pr_xgb2, "auc") 
(auc_xgb2 <- auc_xgb2@y.values[[1]]) #0.957

#plot(prf_xgb2, main="ROC curve with XGboost Linear model")
#text(0.5, 0.5, paste0("auc:",round(auc_xgb2, 4)), cex=1, col="red")

invisible(gc())


######################################## 
print("04. Model evaluation end ----------------")
print(paste0("auc 2:", round(auc_xgb2, 4)))

######################################## 


## Subminssion prep --------------------------------------------------------------------------

######################################## 
print("05 Subminssion prep start ----------------")
######################################## 
sub2 <- fread("../data/ad.tracking/sample_submission.csv")


######################################## 
print("05 Subminssion prep end ----------------")
print(head(sub2))

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
invisible(gc())


######################################## 
print("06 Test data preprocessing end ----------------")
print(head(test, 10))

######################################## 



## Prediction --------------------------------------------------------------------------------

######################################## 
print("07 Prediction start ----------------")
######################################## 

p.test2 <- predict(m_xgb2, test, type = "prob")[,2]
sub2$is_attributed = p.test2

invisible(gc())


######################################## 
print("07 Prediction end ----------------")
print(head(sub2))
######################################## 


## Create submission file --------------------------------------------------------------------

######################################## 
print("08 fwrite start ----------------")
######################################## 

fwrite(sub2, "sub_xgb2_R_10m.csv")

######################################## 
print("08 fwrite end ----------------")
######################################## 


