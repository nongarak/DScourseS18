#Alex Nongard
#PS 10

library(nnet)
library(kknn)
library(e1071)
library(rpart)
library(mlr)
library(xtable)

#############################
## SET UP CODE FROM RANSOM ##
#############################

set.seed(100)



income <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")

names(income) <- c("age","workclass","fnlwgt","education","education.num","marital.status","occupation","relationship","race","sex","capital.gain","capital.loss","hours","native.country","high.earner")



# From UC Irvine's website (http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names)

#   age: continuous.

#   workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.

#   fnlwgt: continuous.

#   education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.

#   education-num: continuous.

#   marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.

#   occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.

#   relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.

#   race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.

#   sex: Female, Male.

#   capital-gain: continuous.

#   capital-loss: continuous.

#   hours-per-week: continuous.

#   native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.



######################

# Clean up the data

######################

# Drop unnecessary columns

income$native.country <- NULL

income$fnlwgt         <- NULL

# Make sure continuous variables are coded as such

income$age            <- as.numeric(income$age)

income$hours          <- as.numeric(income$hours)

income$education.num  <- as.numeric(income$education.num)

income$capital.gain   <- as.numeric(income$capital.gain)

income$capital.loss   <- as.numeric(income$capital.loss)

# Combine levels of categorical variables that currently have too many levels

levels(income$education) <- list(Advanced = c("Masters,","Doctorate,","Prof-school,"), Bachelors = c("Bachelors,"), "Some-college" = c("Some-college,","Assoc-acdm,","Assoc-voc,"), "HS-grad" = c("HS-grad,","12th,"), "HS-drop" = c("11th,","9th,","7th-8th,","1st-4th,","10th,","5th-6th,","Preschool,"))

levels(income$marital.status) <- list(Married = c("Married-civ-spouse,","Married-spouse-absent,","Married-AF-spouse,"), Divorced = c("Divorced,","Separated,"), Widowed = c("Widowed,"), "Never-married" = c("Never-married,"))

levels(income$race) <- list(White = c("White,"), Black = c("Black,"), Asian = c("Asian-Pac-Islander,"), Other = c("Other,","Amer-Indian-Eskimo,"))

levels(income$workclass) <- list(Private = c("Private,"), "Self-emp" = c("Self-emp-not-inc,","Self-emp-inc,"), Gov = c("Federal-gov,","Local-gov,","State-gov,"), Other = c("Without-pay,","Never-worked,","?,"))

levels(income$occupation) <- list("Blue-collar" = c("?,","Craft-repair,","Farming-fishing,","Handlers-cleaners,","Machine-op-inspct,","Transport-moving,"), "White-collar" = c("Adm-clerical,","Exec-managerial,","Prof-specialty,","Sales,","Tech-support,"), Services = c("Armed-Forces,","Other-service,","Priv-house-serv,","Protective-serv,"))

#Alex contribution
#income$high.earner <- as.numeric(income$high.earner)
#Back to Ransosm

# Break up the data:

n <- nrow(income)

train <- sample(n, size = .8*n)

test  <- setdiff(1:n, train)

income.train <- income[train,]

income.test  <- income[test, ]


####################
## MY CODE STARTS ##
####################


View(income)


#5) Setting up the data

## The Classification task

# Define the task:
theTask <- makeClassifTask(data = income.train, target = "high.earner")
print(theTask)

# 3-fold CV strategy:
resampleStrat <- makeResampleDesc(method = "CV", iters = 3)
print(resampleStrat)

# Tuning strategy - random with 10 guesses:
tuneMethod <- makeTuneControlRandom(maxit = 10L)

# Learner predictive algorithms:
#Tree model
predTree <- makeLearner("classif.rpart", 
                        predict.type = "response")
#Logistic Regression
predLog <- makeLearner("classif.glmnet", 
                       predict.type = "response")
#Neural Network
predNN <- makeLearner("classif.nnet", 
                      predict.type = "response")
#Naive Bayes
predNBayes <- makeLearner("classif.naiveBayes", 
                          predict.type = "response")
#kNN
predkNN <- makeLearner("classif.kknn", 
                       predict.type = "response")
#SVM
predSVM <- makeLearner("classif.svm", 
                       predict.type = "response")


#6) Setting up hyperparameters for each algorithm

#Tree model
treeParams <- makeParamSet(makeIntegerParam("minsplit",lower=10,upper=50),
                           makeIntegerParam("minbucket",lower=5,upper=50),
                           makeNumericParam("cp",lower=.001,upper=.2))
#Logistic Regression
logParams <- makeParamSet(makeNumericParam("lambda",lower=0,upper=3),
                          makeNumericParam("alpha",lower=0,upper=1))
#Neural Network
nnParams <- makeParamSet(makeIntegerParam("size",lower=1,upper=10),
                         makeNumericParam("decay",lower=.1,upper=.5),
                         makeIntegerParam("maxit",lower=1000,upper=1000))
#Naive Bayes
#No tuning by nature of the model
#kNN
kNNParams <- makeParamSet(makeIntegerParam("k", lower=1, upper=30))
#SVM (default is Radial)
svmParams <- makeParamSet(makeDiscreteParam("kernel", values = c(2^-2,2^-1,2^0,2^1,2^2,2^10)),
                          makeDiscreteParam("cost", values = c(2^-2,2^-1,2^0,2^1,2^2,2^10)),
                          makeDiscreteParam("gamma", values = c(2^-2,2^-1,2^0,2^1,2^2,2^10)))


#7) Tuning the models

#Tree Model
tunedTree <- tuneParams(learner = predTree,
                         task = theTask,
                         resampling = resampleStrat,
                         measures = list(gmean, f1), 
                         par.set = treeParams,
                         control = tuneMethod,
                         show.info = TRUE)
#Logistic Regression
tunedLog <- tuneParams(learner = predLog,
                       task = theTask,
                       resampling = resampleStrat,
                       measures = list(gmean, f1), 
                       par.set = logParams,
                       control = tuneMethod,
                       show.info = TRUE)
#Neural Network
tunedNN <- tuneParams(learner = predNN,
                      task = theTask,
                      resampling = resampleStrat,
                      measures = list(gmean, f1), 
                      par.set = nnParams,
                      control = tuneMethod,
                      show.info = TRUE)
#Naive Bayes
#No tuning by nature of the model
#kNN
tunedkNN <- tuneParams(learner = predkNN,
                       task = theTask,
                       resampling = resampleStrat,
                       measures = list(gmean, f1), 
                       par.set = kNNParams,
                       control = tuneMethod,
                       show.info = TRUE)
#SVM
tunedSVM <- tuneParams(learner = predSVM,
                       task = theTask,
                       resampling = resampleStrat,
                       measures = list(gmean, f1), 
                       par.set = svmParams,
                       control = tuneMethod,
                       show.info = TRUE)


#8) Optimization, training, prediction, and assessment

#Tree model
# Apply the optimal algorithm parameters to the model
predTree <- setHyperPars(learner=predTree, par.vals = tunedTree$x)
# Verify performance on cross validated sample sets
resample(predTree,theTask,resampleStrat,measures=list(gmean,f1))
# Train the final model
finalTree <- train(learner = predTree, task = theTask)
# Predict in test set
Treeprediction <- predict(finalTree, newdata = income.test)
# Performance
print(performance(Treeprediction, measures = list(gmean,f1)))

#Logistic Regression
# Apply the optimal algorithm parameters to the model
predLog <- setHyperPars(learner=predLog, par.vals = tunedLog$x)
# Verify performance on cross validated sample sets
resample(predLog,theTask,resampleStrat,measures=list(gmean,f1))
# Train the final model
finalLog <- train(learner = predLog, task = theTask)
# Predict in test set
Logprediction <- predict(finalLog, newdata = income.test)
# Performance
print(performance(Logprediction, measures = list(gmean,f1)))

#Neural Network
# Apply the optimal algorithm parameters to the model
predNN <- setHyperPars(learner=predNN, par.vals = tunedNN$x)
# Verify performance on cross validated sample sets
resample(predNN,theTask,resampleStrat,measures=list(gmean,f1))
# Train the final model
finalNN <- train(learner = predNN, task = theTask)
# Predict in test set
NNprediction <- predict(finalNN, newdata = income.test)
# Performance
print(performance(NNprediction, measures = list(gmean,f1)))

#Naive Bayes
# No tuning by the nature of the model
# Train the final model
finalNBayes <- train(learner = predNBayes, task = theTask)
# Predict in test set
NBayesprediction <- predict(finalNBayes, newdata = income.test)
# Performance
print(performance(NBayesprediction, measures = list(gmean,f1)))


#kNN
# Apply the optimal algorithm parameters to the model
predkNN <- setHyperPars(learner=predkNN, par.vals = tunedkNN$x)
# Verify performance on cross validated sample sets
resample(predkNN,theTask,resampleStrat,measures=list(gmean,f1))
# Train the final model
finalkNN <- train(learner = predkNN, task = theTask)
# Predict in test set
kNNprediction <- predict(finalkNN, newdata = income.test)
# Performance
print(performance(kNNprediction, measures = list(gmean,f1)))

#SVM
# Apply the optimal algorithm parameters to the model
predSVM <- setHyperPars(learner=predSVM, par.vals = tunedSVM$x)
# Verify performance on cross validated sample sets
resample(predSVM,theTask,resampleStrat,measures=list(gmean,f1))
# Train the final model
finalSVM <- train(learner = predSVM, task = theTask)
# Predict in test set
SVMprediction <- predict(finalSVM, newdata = income.test)
# Performance
print(performance(SVMprediction, measures = list(gmean,f1)))


#9)
#Create objects to be put into LaTeX table
modeltype<- c("Tree","Logarithmic","NN","NBayes","kNN","SVM")
GMean<- c(print(performance(Treeprediction, measures = list(gmean))),
          print(performance(Logprediction, measures = list(gmean))),
          print(performance(NNprediction, measures = list(gmean))),
          print(performance(NBayesprediction, measures = list(gmean))),
          print(performance(kNNprediction, measures = list(gmean))),
          print(performance(SVMprediction, measures = list(gmean))))
F1<- c(print(performance(Treeprediction, measures = list(f1))),
       print(performance(Logprediction, measures = list(f1))),
       print(performance(NNprediction, measures = list(f1))),
       print(performance(NBayesprediction, measures = list(f1))),
       print(performance(kNNprediction, measures = list(f1))),
       print(performance(SVMprediction, measures = list(f1))))
performance<- data.frame(modeltype, GMean, F1)

#Use XTable to create a LaTeX table
performance<- xtable(performance)
print.xtable(performance)
