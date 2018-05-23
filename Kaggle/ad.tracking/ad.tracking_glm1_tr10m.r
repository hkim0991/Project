# ---
# title: "ad.tracking_modeling"
# author: "kimi"
# date: "2018?ÖÑ 5?õî 18?ùº"
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
rm(train_sample_path)
train_path <- "../data/ad.tracking/train_10m.csv"
#train_path <- "../data/ad.tracking/train.csv"
test_path  <- "../data/ad.tracking/test.csv"

#subm_path <- "../data/ad.tracking/sample_submission.csv"
#train_sample_path <- "../data/ad.tracking/train_sample.csv"

train <- fread(train_path, showProgress = T, stringsAsFactors = F, na.string=c("", NA))
test <- fread(test_path, showProgress = T, stringsAsFactors = F)

# subm <- fread(subm_path, showProgress = F, stringsAsFactors = F)
# df <- fread(train_sample_path, showProgress = F, stringsAsFactors = F, na.string=c("", NA))

#set.seed(0)
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

tr2$is_attributed <- as.factor(tr2$is_attributed)

X<- copy(tr2[, -6])
y <- tr2$is_attributed

set.seed(0)
tri <- createDataPartition(y, p=0.7, list=F)

X <- X[tri,]
y <- y[tri]

X_val <- X[-tri,]
y_val <- y[-tri]


######################################## 
print("02 Data partition end ----------------")
print(dim(X))
print(dim(X_val))

######################################## 


## Modeling: Logistic Regression ---------------------------------------------------------------

######################################## 
print("03 Modeling - glm start ----------------")
######################################## 

m_glm<- glm(y~ ., data=X, family="binomial")

######################################## 
print("03 Modeling - glm end ----------------")
print(summary(m_glm))
######################################## 


# Model Evaludation ----------------------------------------------------------------------------

######################################## 
print("04. Model evaluation start ----------------")
######################################## 


#install.packages("ROCR")
library(ROCR)

p.valid <- predict(m_glm, X_val, type = "response")

#pr_t <- prediction(p.train, y)
#prf_t <- performance(pr_t, "tpr", "fpr")
#auc_t <- performance(pr_t, "auc") # 0.928
#(auc_t <- auc_t@y.values[[1]]) 

pr <- prediction(p.valid, y_val)
prf <- performance(pr, "tpr", "fpr")
auc <- performance(pr, "auc") # 0.939
(auc <- auc@y.values[[1]]) 

plot(prf)
title("ROC graph")
text(0.15, 0.8, round(auc, 3), cex=1, col="red")


######################################## 
print("04. Model evaluation end ----------------")
print(round(auc, 3))
######################################## 


## Subminssion prep --------------------------------------------------------------------------

######################################## 
print("05 Subminssion prep start ----------------")
######################################## 

sub <- data.table(click_id = test$click_id, is_attributed = NA)
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
print(head(test, 10))
######################################## 


## Prediction --------------------------------------------------------------------------------

######################################## 
print("07 Prediction start ----------------")
######################################## 

p.test <- predict(m_glm, test, type = "response")
sub$is_attributed = p.test

######################################## 
print("07 Prediction end ----------------")
######################################## 


######################################## 
print("08 fwrite start ----------------")
######################################## 

fwrite(sub, "sub_glm_R_10m.csv")


######################################## 
print("08 fwrite end ----------------")
######################################## 







