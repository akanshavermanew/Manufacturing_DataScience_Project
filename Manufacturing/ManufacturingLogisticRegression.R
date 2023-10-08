#Manufacturing project is build using simple logistic regression and here we have to calculate and submit hard classes i.e cutoff also i.e KS
#Till probability scores it is correct not able to do cutoff in this code

library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(tidyr)
library(ROCit)

setwd("F:/R study materials/Projects/Manufacturing")
## ----
pts_train=read.csv("product_train.csv",stringsAsFactors = FALSE)
pts_test=read.csv("product_test.csv",stringsAsFactors = FALSE)

glimpse(pts_train)

vis_dat(pts_train)
unique(pts_train$went_on_backorder )

#Since we will be using random forest we need to convert data type of response (which is store in this case) to factor type using function as.factor. This is how randomforest differentiates from regression & classification.If we need to build a regression model then response variable should be kept numeric else factor for classification.

backorder_func=function(x){
  x=ifelse(pts_train$went_on_backorder=="Yes",1,0)
  x=as.factor(x)
  
  return(x)
}
#for classification if we are using random forest algo then we need to convert the output variable into factor 
#and for linear regression we need to make it as numeric
#But here the output variable i.e. 'y' is already factor so there is no need to convert the outcome variable to factor5

#pts_train$went_on_backorder =as.factor(as.numeric(pts_train$went_on_backorder))

dp_pipe=recipe(went_on_backorder~ .,data=pts_train) %>% 
  update_role(sku,new_role = "drop_vars") %>%
  update_role(potential_issue,deck_risk,oe_constraint,ppap_risk,stop_auto_buy,rev_stop,new_role="to_dummies") %>% 
  step_rm(has_role("drop_vars")) %>%
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>% 
  step_mutate_at(went_on_backorder,fn=backorder_func,skip=TRUE) %>% 
  step_impute_median(all_numeric(),-all_outcomes())

dp_pipe=prep(dp_pipe)

train=bake(dp_pipe,new_data = NULL)
test=bake(dp_pipe,new_data=pts_test)

vis_dat(train)
glimpse(train)

summary(dp_pipe)

set.seed(2)
s=sample(1:nrow(train),0.8*nrow(train))
t1=train[s,]
t2=train[-s,]

## remove vars with vif higher than 10
for_vif=lm(went_on_backorder~.,data=t1)
summary(for_vif)   

for_vif=lm(went_on_backorder~.-deck_risk_X__other__ -ppap_risk_X__other__
           -stop_auto_buy_X__other__ -forecast_6_month -sales_6_month -sales_9_month 
           -forecast_9_month -sales_1_month -perf_12_month_avg  -min_bank         ,
           data=t1)

# we are using this strictly for vif values only
# this model has nothing to do with the classification problem

sort(vif(for_vif),decreasing = T)[1:3]

summary(for_vif)




log_fit=glm(went_on_backorder~.-deck_risk_X__other__ - -ppap_risk_X__other__
            -ppap_risk_X__other__ -stop_auto_buy_X__other__ -forecast_6_month 
            -forecast_3_month -sales_3_month -sales_6_month -sales_1_month 
            -forecast_9_month-perf_12_month_avg          ,
            data=t1,
            family = "binomial")
summary(log_fit)
sort(vif(log_fit),decreasing = T)[1:3]

log_fit=stats::step(log_fit)


####


summary(log_fit)

formula(log_fit)

log_fit=glm(went_on_backorder ~ national_inv + lead_time + in_transit_qty + 
              min_bank + pieces_past_due + perf_6_month_avg + local_bo_qty + 
              deck_risk_Yes + rev_stop_X__other__,
            data=t1,family="binomial")

summary(log_fit)

#### performance on t2 with auc score

val.score=predict(log_fit,newdata = t2,type='response')

pROC::auc(pROC::roc(t2$went_on_backorder,val.score))

### now fitting model on the entire data

for_vif=lm(went_on_backorder~.-deck_risk_X__other__-ppap_risk_X__other__  
           -stop_auto_buy_X__other__ -forecast_6_month  -sales_6_month -sales_9_month 
           -forecast_9_month -sales_1_month -perf_12_month_avg -min_bank           ,data=train)

sort(vif(for_vif),decreasing=T)[1:3]

summary(for_vif)

log_fit.final=glm(went_on_backorder~.-deck_risk_X__other__-ppap_risk_X__other__  
                  -stop_auto_buy_X__other__ -forecast_6_month  -sales_6_month -sales_9_month 
                  -forecast_9_month -sales_1_month -perf_12_month_avg -min_bank  ,data=train,family = "binomial")

summary(log_fit.final)

log_fit.final=stats::step(log_fit.final)

summary(log_fit.final)

formula(log_fit.final)

log_fit.final=glm(went_on_backorder ~ national_inv + lead_time + in_transit_qty + 
                    forecast_3_month + pieces_past_due + perf_6_month_avg + local_bo_qty + 
                    potential_issue_X__other__ + deck_risk_Yes + ppap_risk_Yes + 
                    rev_stop_X__other__ , data=train,
                  family="binomial")
summary(log_fit.final)


### finding cutoff for hard classes

train.score=predict(log_fit.final,newdata = train,type='response')

real=train$went_on_backorder


variable=predict(log_fit.final,train,type="response")

test.prob.score= as.numeric(predict(log_fit.final,newdata = test,type='response')>0.3)
test.prob.score=ifelse(test.prob.score==1,'Yes','No')
write.table(test.prob.score,"Aswini_Banking_P5_part2.csv",row.names = F,col.names="y")
#####
library(ROCR)
RP=prediction(variable,train$went_on_backorder)
RPE=performance(RP,"tpr","fpr")
plot(RPE,colorize=T,print.cutoffs.at=seq(0.1,by=0.1))

# 0.3 THRESHOLD OBTAINED FROM GRAPH
#CONFUSION MATRIX
table(ActualValue=t2$went_on_backorder,PredictedValue=val.score>0.1)

ks=(0/(0+333))-(4/(4+49679     ))
round(ks,2)
#0.47

library(ggplot2)
bank_train$score=predict(log_fit_final,bank_train,type="response")
ggplot(bank_train,aes(x=score,y=y,color=factor(y)))+geom_point()+geom_jitter()

k=read.csv("Aswini_Banking_P5_part2.csv")
table(k$y)
