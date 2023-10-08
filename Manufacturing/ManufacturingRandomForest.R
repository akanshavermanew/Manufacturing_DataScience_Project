#Manufacturing project is build using Random forest and here we have to calculate and submit hard classes i.e cutoff also i.e KS
#KS score is 0.05

library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(tidyr)
library(ROCit)
library(vip)
library(rpart.plot)
library(DALEXtra)

setwd("F:/R study materials/Projects/Manufacturing")
## ----
pt_train=read.csv("product_train.csv",stringsAsFactors = FALSE)
pt_test=read.csv("product_test.csv",stringsAsFactors = FALSE)

glimpse(pt_train)

vis_dat(pt_train)
length(unique(pt_train$sku ))


backorder_func=function(x){
  x=ifelse(pt_train$went_on_backorder=="Yes",1,0)
  x=as.factor(x)
  
  return(x)
}
#Since we will be using random forest we need to convert data type of response (which is store in this case) to factor type using function as.factor. This is how randomforest differentiates from regression & classification.If we need to build a regression model then response variable should be kept numeric else factor for classification.


#for classification if we are using random forest algo then we need to convert the output variable into factor 
#and for linear regression we need to make it as numeric
#But here the output variable i.e. 'y' is already factor so there is no need to convert the outcome variable to factor5

#pt_train$went_on_backorder =as.factor(pt_train$went_on_backorder)

dp_pipe=recipe(went_on_backorder~ .,data=pt_train) %>% 
  update_role(sku,new_role = "drop_vars") %>%
  update_role(potential_issue,deck_risk,oe_constraint,ppap_risk,stop_auto_buy,rev_stop,new_role="to_dummies") %>% 
  step_rm(has_role("drop_vars")) %>%
  step_mutate_at(went_on_backorder,fn=backorder_func,skip=TRUE) %>% 
  
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>% 
  step_impute_median(all_numeric(),-all_outcomes())

dp_pipe=prep(dp_pipe)

train=bake(dp_pipe,new_data = NULL)
test=bake(dp_pipe,new_data=pt_test)

vis_dat(train)
glimpse(train)

summary(dp_pipe)

#building random forest
rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

folds = vfold_cv(train, v = 3)

rf_grid = grid_regular(mtry(c(5,15)), trees(c(100,500)),     #for random forest passing some values is compulsory otherwise it will give error in decision tree the case was not like this
                       min_n(c(2,10)),levels = 3)

#IMP BELOW 2 STEPS
# c(5,19)  means start with 5 and go till 19
# mtry values should be <= features in your table
my_res=tune_grid(
  rf_model,
  went_on_backorder~.,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
)


autoplot(my_res)+theme_light()

fold_metrics=collect_metrics(my_res)    #grid search it will show table with parameters like mtry,tree_depth,etc (parameters of whatever model you are using

my_res %>% show_best()            #it will give best roc_auc value for our model from grid serch with all parameters of whatever model you are using like mtry,tree_depth,etc

#Finalizing the model
final_rf_fit=rf_model %>% 
  set_engine("ranger",importance='permutation') %>% 
  finalize_model(select_best(my_res,"roc_auc")) %>% 
  fit(went_on_backorder~.,data=train)

# variable importance 

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))


# predicitons(first check cutoff and then do predictions on test data)

train_pred=predict(final_rf_fit,new_data = train,type="prob")%>% select(.pred_1)

#do these 3 steps after finding cutoff
test_pred=predict(final_rf_fit,new_data = test,type="prob")%>% select(.pred_1)
a=as.numeric(test_pred>0.05)
test_preds=ifelse(a==1,'Yes','No')
#(0.05 is ks)

### finding cutoff for hard classes

train.score=train_pred$.pred_1

real=train$went_on_backorder

# KS plot

rocit = ROCit::rocit(score = train.score, 
                     class = real) 

kplot=ROCit::ksplot(rocit)

# cutoff on the basis of KS

my_cutoff=kplot$`KS Cutoff`

Score=1-(0.025/0.0507)
## test hard classes 

test_hard_class=as.numeric(test_preds>my_cutoff)

#write csv
write.table(test_preds,"Akansha_Verma_P3_part2.csv",row.names=F,col.names='went_on_backorder')
getwd()





#Quiz
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#q1)Find the difference between median of variables perf_6_month_avg and perf_12_month_avg when your target values is Yes
  
filtered_df <- pts_train[pts_train$went_on_backorder == "Yes", ]
median_6_month <- median(filtered_df$perf_6_month_avg)
median_12_month <- median(filtered_df$perf_12_month_avg)
difference <- median_6_month - median_12_month
#ans - 0
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
#q2)Do a chisq test to see whether variable deck_risk affects your target variable. Write down p-value only, for the test here. Round off your answer to 4 decimal digits.
  
contingency_table <- table(pts_train$deck_risk, pts_train$went_on_backorder )
chi_square_test <- chisq.test(contingency_table)
p_value <- chi_square_test$p.value
rounded_p_value <- round(p_value, 4)
#ans - 0.0012

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#q3)Find out the % zeros in the variable pieces_past_due . Round off your answer to two decimal digits
  
zero_percentage <- mean(pts_train$pieces_past_due == 0) * 100
rounded_zero_percentage <- round(zero_percentage, 2)
#ans - 99.13

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#q4)Find the correlation coefficient between forecast_9_month and sales_9_month. Round it off to two decimal digits.
  
correlation <- cor(pts_train$forecast_9_month, pts_train$sales_9_month)
rounded_correlation <- round(correlation, 2)
#ans - 0.92

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#q5)Is average min_bank significantly different across two target categories. Do a t-test and conclude according to p-value(answer should be the p-value only),round it off to 2 decimal places. Consider alpha=0.05.
  
min_bank_target1 <- pts_train$min_bank[pts_train$went_on_backorder  == "Yes"]
min_bank_target2 <- pts_train$min_bank[pts_train$went_on_backorder  == "No"]
t_test_result <- t.test(min_bank_target1, min_bank_target2, var.equal = FALSE)
p_value <- t_test_result$p.value
rounded_p_value <- round(p_value, 2)
if (rounded_p_value < 0.05) {
  conclusion <- "The average min_bank is significantly different across two target categories."
} else {
  conclusion <- "The average min_bank is not significantly different across two target categories."
}

#ans - 0.62

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#q6)Which function can be used to remove objects from memory?
  #ans - rm
  
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#q7)If there are 30 predictor variables in the data being considered for classification with function randomForest , what would be default value of argument mtry.
#ans - 5
#explaination - In the randomForest function in R, the default value of the mtry argument is the square root of the total number of predictor variables.
#Since you mentioned that there are 30 predictor variables in your data, the default value of mtry would be the square root of 30, which is approximately 5.48. However, since mtry needs to be an integer, R will automatically round down the square root value to the nearest integer. Therefore, the default value of mtry in this case would be 5.
  
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  
  
#q8)Which among these aesthetics can be mapped to both categorical and numerical data?
#ans - color
--------------------------------------