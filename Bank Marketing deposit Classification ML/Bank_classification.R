#-----------------------
#----Import Packages----
#-----------------------
install.packages("readr")
library(readr)
install.packages("dplyr")
library(dplyr)
install.packages("datawizard")
library(datawizard)
install.packages("caret")
library(caret)
install.packages("tidyverse")
library(tidyverse)
install.packages("randomForest")
library(randomForest)
install.packages("ggplot2")
library(ggplot2)

#---------------------
#----Read data set---
#--------------------
bank<- data.frame(read_csv("bank.csv"))
head(bank,10)

#----------------------
#----About data set----
#----------------------
# structure
str(bank)

# statistical summary of data set
summary(bank)

# dimension row,col of data set
dim(bank)

# column names of data set
colnames(bank)

#---------------------
#----Data Cleaning----
#---------------------
#remove duplicates
bank <- unique(bank)
  
#check for null values
colSums(is.na(bank))

#Function to Replace numeric columns null with median , categorical with mode
replace_mode <- function(x) {
  names(sort(table(x), decreasing = TRUE))[1]
}

#Replace missing values
for(col in names(bank)){
  if(is.numeric(bank[[col]])){
    bank[[col]][is.na(bank[[col]])] <- median(bank[[col]], na.rm = TRUE)
  } else {
    bank[[col]][is.na(bank[[col]])] <- replace_mode(bank[[col]])
  }
}

#check for outliers
boxplot(bank[sapply(bank, is.numeric)])

#replace extreme outliers with median to avoid bias 
bank <- bank %>% mutate(balance=replace(balance,balance>40000,median(balance,na.rm = TRUE)))

# remove columns with no data (contact,poutcome) are less useful
bank <- bank %>% select(-contact, -poutcome)

#---------------------
#----Pre Processing---
#---------------------

#convert categorical data to factor
for(col in names(bank)){
  if(is.character(bank[[col]])){
    bank[[col]]=as.factor(bank[[col]])
  }
}

#split data to train/test 
set.seed(123)
split <- data_partition(bank,proportion = 0.8)
train_data <- split$p_0.8
test_data <- split$test
str(train_data)

#Encode 
train_data$deposit <- as.factor(train_data$deposit)
test_data$deposit  <- as.factor(test_data$deposit)

dummy <- dummyVars(deposit ~ ., data = train_data)

xtrain <- as.data.frame(predict(dummy,newdata= train_data))
xtest <- as.data.frame(predict(dummy,newdata= test_data))
ytrain <- train_data$deposit
ytest <- test_data$deposit

#Remove generated index Id to prevent model confusion or affect predicition
xtrain$.row_id <- NULL
xtest$.row_id  <- NULL

#Scale
scaler <- preProcess(xtrain, method = c("center", "scale"))
xtrain <- predict(scaler, xtrain)
xtest  <- predict(scaler, xtest)

#------------------------
#----Model & evaluation--
#------------------------

#Model
model <- randomForest(x = xtrain, y = ytrain, ntree = 500 , importance = TRUE)
pred <- predict(model, xtest)

result <- data.frame(
  Actual = ytest,
  Predicted = pred,
  Correct = ifelse(ytest == pred, "Yes", "No")
)

#Evaluation using confusion matrix
cm <- confusionMatrix(pred, ytest)

#Accuracy 
cat("Accuracy:", cm$overall["Accuracy"], "\n")

# Precision, Recall
cm$byClass 

# feature importance
imp<- importance(model)
varImpPlot(model)

#---------------------
#----Visualization----
#---------------------

#1.Target Variable Distribution (Deposit Class Balance)
#This graph shows the distribution of 
#customers who subscribed vs not subscribed.
#It helps identify whether the data set is 
#balanced or imbalanced.
ggplot(bank, aes(x = deposit)) +geom_bar(fill = "maroon") +theme_minimal() +
  labs(title = "Distribution of Deposit (Target Variable)",
       x = "Deposit (Yes/No)",y = "Count")

#2. Age Distribution by Deposit Outcome
#This shows how age influences subscription behavior. It helps identify which
#age groups are more likely to subscribe.
ggplot(bank, aes(x = age, fill = deposit)) +
  geom_histogram(bins = 30, alpha = 0.6, position = "identity") +
  theme_minimal() +
  labs(title = "Age Distribution by Deposit Outcome",
       x = "Age",y = "Frequency")

#3. Balance vs Deposit (Box Plot)
#This plot compares account balances between customers who subscribed
#and those who did not. It helps detect financial influence on decisions.
ggplot(bank, aes(x = deposit, y = balance, fill = deposit)) +
  geom_boxplot() +theme_minimal() +
  labs(title = "Account Balance vs Deposit",
       x = "Deposit",y = "Balance")

#4. Confusion Matrix Heatmap (Model Performance)
#This visualization shows how well the Random Forest model correctly and
#incorrectly classified customers.
cm_df <- as.data.frame(cm$table)

ggplot(cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "lightblue") +
  theme_minimal() +
  labs(title = "Confusion Matrix Heatmap",
       x = "Predicted",
       y = "Actual")


# 5. Feature Importance (Random Forest Insight)
#This shows which variables have the most influence on predicting whether a 
#customer will subscribe to a term deposit.
importance_df <- data.frame(
  Feature = rownames(imp),
  Importance = imp[,1]
)

ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Feature Importance (Random Forest)",
       x = "Features",
       y = "Importance")


#---------------------
#----Interpretation---
#---------------------
#The Balanced Accuracy of 85.2% confirms that the model is robust and not biased by the 
#slight class imbalance in the data set. The Random Forest successfully captured non-linear
#relationships between customer demographics and financial behavior to provide actionable 
#business intelligence.

