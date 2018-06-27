
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
print("03 Modeling - XGboost start ----------------")
######################################## 

# Setting up train controls
repeats <- 3
numbers <- 10
tunel <- 10


set.seed(1234)
ctrl1 = trainControl(method="repeatedcv",
                     number=numbers,
                     repeats = repeats,
                     classProbs=TRUE,
                     summaryFunction = twoClassSummary)

grid1 = expand.grid(nrounds = 100, 
                    max_depth = 7,
                    eta = 0.2,
                    gamma = 0,
                    min_child_weight = 3,
                    colsample_bytree = 0.5,
                    subsample = 0.7)

# model 1
m_xgb1 <- train(is_attributed~., data=train, 
                method = "xgbTree",
                metric="ROC",
                trControl=ctrl1,
                tuneLength=tunel,
                nthread = 4,
                tuneGrid = grid1)


#plot(m_dt1$finalModel, main="Classification Tree")
#text(m_dt1$finalModel)



######################################## 
print("03 Modeling -  XGboost end ----------------")
print(m_xgb1)

######################################## 


# Model Evaludation ----------------------------------------------------------------------------

######################################## 
print("04. Model prediction & evaluation start ----------------")
######################################## 

p.valid1 <- predict(m_xgb1, valid, type = "prob")

#install.packages("ROCR")
library(ROCR)


# model 1
pr_xgb1 <- prediction(p.valid1[, 2], valid$is_attributed)
prf_xgb1 <- performance(pr_xgb1, "tpr", "fpr")
auc_xgb1 <- performance(pr_xgb1, "auc") 
(auc_xgb1 <- auc_xgb1@y.values[[1]]) #0.958

#plot(prf_xgb2, main="ROC curve with XGboost Linear model")
#text(0.5, 0.5, paste0("auc:",round(auc_xgb2, 4)), cex=1, col="red")

######################################## 
print("04. Model evaluation end ----------------")
print(paste0("auc 1:", round(auc_xgb1, 4)))

######################################## 


## Subminssion prep --------------------------------------------------------------------------

######################################## 
print("05 Subminssion prep start ----------------")
######################################## 

#sub1 <- data.table(click_id = test$click_id, is_attributed = NA)
#sub2 <- data.table(click_id = test$click_id, is_attributed = NA)

sub1 <- fread("../data/ad.tracking/sample_submission.csv")


#test$click_id <- NULL


######################################## 
print("05 Subminssion prep end ----------------")
print(head(sub1))
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
print(head(test, 10))

######################################## 



## Prediction --------------------------------------------------------------------------------

######################################## 
print("07 Prediction start ----------------")
######################################## 

p.test1 <- predict(m_xgb1, test, type = "prob")[,2]
sub1$is_attributed = p.test1


######################################## 
print("07 Prediction end ----------------")
print(head(sub1))
######################################## 


## Create submission file --------------------------------------------------------------------

######################################## 
print("08 fwrite start ----------------")
######################################## 

fwrite(sub1, "sub_xgb1_R_10m.csv")

######################################## 
print("08 fwrite end ----------------")
######################################## 