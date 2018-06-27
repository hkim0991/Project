# ---
# title: "ad.tracking_modeling"
# author: "kimi"
# date: "2018??Ä? 5??ç° 18??î™"
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
#train <- train[sample(.N, 1e5), ]

invisible(gc())



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
tri <- createDataPartition(y=tr2$is_attributed, p=0.70, list=FALSE)

train <- tr2[tri,]
valid <- tr2[-tri,]

# Setting levels for both train/valid data
levels(train$is_attributed) <- make.names(levels(factor(train$is_attributed)))
levels(valid$is_attributed) <- make.names(levels(factor(valid$is_attributed)))

invisible(gc())


######################################## 
print("02 Data partition end ----------------")
print(dim(train))
print(dim(valid))
######################################## 


## Modeling: Decision Tree --------------------------------------------------------------

######################################## 
print("03 Modeling - Decision Tree start ----------------")
######################################## 

# Setting up train controls
library(rpart)
m_dt1 <- rpart(is_attributed~., data=train)

#plot(m_dt1$finalModel, main="Classification Tree")
#text(m_dt1$finalModel)

invisible(gc())


######################################## 
print("03 Modeling -  Decision Tree end ----------------")
print(plot(m_dt1))

######################################## 


# Model Evaludation ----------------------------------------------------------------------------

######################################## 
print("04. Model prediction & evaluation start ----------------")
######################################## 

p.valid1 <- predict(m_dt1, valid, type = "prob")
head(p.valid1)

#install.packages("ROCR")
library(ROCR)

pr_dt1 <- prediction(p.valid1[, 2], valid$is_attributed)
prf_dt1 <- performance(pr_dt1, "tpr", "fpr")
auc_dt1 <- performance(pr_dt1, "auc") 
(auc_dt1 <- auc_dt1@y.values[[1]]) #0.910

#plot(prf_dt1, main="ROC curve with decision tree model")
#text(0.5, 0.5, paste0("auc:",round(auc_dt1, 3)), cex=1, col="red")


invisible(gc())


######################################## 
print("04. Model evaluation end ----------------")
print(paste0("auc 1: ", round(auc_dt1, 3)))

######################################## 


## Subminssion prep --------------------------------------------------------------------------

######################################## 
print("05 Subminssion prep start ----------------")
######################################## 

sub <- fread("../data/ad.tracking/sample_submission.csv")


invisible(gc())
dim(sub)


######################################## 
print("05 Subminssion prep end ----------------")
print(head(sub))

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

#p.test <- predict(m_dt1, data=as.matrix(test[, colnames(test)]), type = "prob")
#p.test <- as.data.frame(p.test)

p.test <- predict(m_dt1, test, type = "prob")[,2]
sub$is_attributed = p.test


invisible(gc())


######################################## 
print("07 Prediction end ----------------")
print(head(sub, 10))

######################################## 



## Create submission file --------------------------------------------------------------------

######################################## 
print("08 fwrite start ----------------")
######################################## 

fwrite(sub, "sub_dt3_rpart_R_10m.csv")

######################################## 
print("08 fwrite end ----------------")
######################################## 


