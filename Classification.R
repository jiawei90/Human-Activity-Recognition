library(keras)
library(ggbiplot)
install.packages("ggbiplot")
getwd()
#work computer
setwd("C:\\Users\\chee.jw\\Dropbox\\NUS\\KE 5206\\CA\\Human activity recognition\\UCI HAR Dataset")
setwd("/home/iss-user/Desktop/human/data")

##### INITIALIZATION #####
#trainData <- read.table("train\\X_train.txt")
#trainPredict <- read.table("train\\y_train.txt")
#testData <- read.table("test\\X_test.txt")
#testPredict <- read.table("test\\y_test.txt")
trainData <- read.table("X_train.txt")
trainPredict <- read.table("y_train.txt")
testData <- read.table("X_test.txt")
testPredict <- read.table("y_test.txt")

mergePredict <- rbind(trainPredict, testPredict)
mergeData <- rbind(trainData, testData)

#To ensure all are dataframe of numeric type
mergePredict <- lapply(mergePredict, as.numeric)
mergePredict <- data.frame(Reduce(rbind,mergePredict))

working <- mergeData
working$predict <- mergePredict
working$predict <- as.numeric(unlist(working$predict))-1

#To separate into train and test datasets
set.seed(5678)
working <- working[sample(nrow(working)),]
split <- floor(nrow(working)/3)

#convert to matrix
workingKeras <- as.matrix(working)
dimnames(workingKeras) <- NULL

testTarget <- workingKeras[0:split,562]
trainTarget <- workingKeras[(split+1):nrow(workingKeras),562]
trainLabels <- to_categorical(trainTarget)
testLabels <- to_categorical(testTarget)

##### PRINCIPLE COMPONENT ANALYSIS #####
testPCA <- working[0:split, 0:561]
trainPCA <- working[(split+1):nrow(working), 0:561]

# conduct PCA on train dataset
prin_comp <- prcomp(trainPCA, scale. = TRUE)
names(prin_comp)
summary(prin_comp)

# Center = Mean, Sale = Standard Deviation of the variables that are used for normalization prior to implementing PCA
# Rotation provides the principal component loading
prin_comp$center
prin_comp$scale
prin_comp$rotation
prin_comp$rotation[1:5,1:4]
dim(prin_comp$x)

# parameter scale = 0 ensures that arrows are scaled to represent the loadings
biplot(prin_comp, scale = 0)

# 2nd variation of biplot
g <- ggbiplot(prin_comp, obs.scale = 1, var.scale = 1, 
              groups = working[(split+1):nrow(working), 562], ellipse = TRUE, 
              circle = TRUE)
g <- g + scale_color_continuous(name = '')
g <- g + theme(legend.direction = 'horizontal', 
               legend.position = 'top')
print(g)

#compute standard deviation of each principal component
std_dev <- prin_comp$sdev
pr_var <- std_dev^2
pr_var[1:10]

#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:20]
sum(prop_varex[1:200])
write.csv(prop_varex, file="Output\\PCA\\Proportion of Variance.csv")
# sum value of first 200 variables = 0.9933
# sum value of first 150 variables = 0.98026

#scree plot
plot(prop_varex, xlab = "Principal Component", ylab = "Proportion of Variance Explained", type = "b")

#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained",
       type = "b")

# train on first 200 Principle components
train.data_PCA <- data.frame(prin_comp$x)
train.data_PCA <- train.data_PCA[,1:200]
str(train.data_PCA)  
rm(list = "trainData")
rm(list = "mergeData", "mergePredict", "testData", "testPredict", "trainPredict")

train.data_PCA <- as.matrix(train.data_PCA)

#transform test into PCA
test.data <- predict(prin_comp, newdata = testPCA)
test.data <- as.data.frame(test.data)
test.data <- test.data[,1:200]
test.data <- as.matrix(test.data)

##### Model 1 MLP w One Hidden Layer sigmoid, 28 #####
rm(list="model_PCA_MLP_1", "startTime_PCA_MLP_1","tensorboard_MLP_1","history_PCA_MLP_1", "time.taken_PCA_MLP_1")
model_PCA_MLP_1 <- keras_model_sequential()
model_PCA_MLP_1 %>% 
  layer_dense(units = 28, activation = 'sigmoid', input_shape = c(200)) %>%  #this is the first hidden layer
  layer_dense(units = 6, activation = 'softmax') 

summary(model_PCA_MLP_1)
get_config(model_PCA_MLP_1)
get_layer(model_PCA_MLP_1,index=1)
model_PCA_MLP_1$layers
model_PCA_MLP_1$inputs
model_PCA_MLP_1$outputs

model_PCA_MLP_1 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

tensorboard_MLP_1 = callback_tensorboard(log_dir = NULL, histogram_freq = 0, batch_size = 5, write_graph = TRUE, 
                                     write_images = TRUE)

startTime_PCA_MLP_1 <- proc.time()
history_PCA_MLP_1 <- model_PCA_MLP_1 %>% fit(
  train.data_PCA, 
  trainLabels, 
  epochs = 200,
  batch_size = 5, 
  validation_split = 0.2,
  callbacks = tensorboard_MLP_1
)
time.taken_PCA_MLP_1 <- proc.time() - startTime_PCA_MLP_1
plot(history_PCA_MLP_1)

# Predict the classes for the test data
classes <- model_PCA_MLP_1 %>% predict_classes(test.data, batch_size = 128)
# Confusion matrix
table(testTarget, classes)
write.table(table(testTarget, classes),file="Output\\confusionMatrix_Model_PCA_MLP_1.txt", sep="\t")

# Evaluate on test data and labels
score <- model_PCA_MLP_1 %>% evaluate(test.data, testLabels, batch_size = 128)
# Print the score
print(score)


##### Model 2 MLP w Two Hidden Layers sigmoid sigmoid 28 28 #####
rm(list="mpdel_PCA_MLP_2", "tensorboard_MLP_2","startTime_PCA_MLP_2","history_PCA_MLP_2")
model_PCA_MLP_2 <- keras_model_sequential() 

model_PCA_MLP_2 %>% 
  layer_dense(units = 28, activation = 'sigmoid', input_shape = c(200)) %>% 
  layer_dense(units = 28, activation = 'sigmoid') %>% 
  layer_dense(units = 6, activation = 'softmax')

model_PCA_MLP_2 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

tensorboard_MLP_2 = callback_tensorboard(log_dir = NULL, histogram_freq = 0, batch_size = 5, write_graph = TRUE, 
                                    write_images = TRUE)

startTime_PCA_MLP_2 <- proc.time()
history_PCA_MLP_2 <- model_PCA_MLP_2 %>% fit(
  train.data_PCA, trainLabels, 
  epochs = 200, batch_size = 5,
  validation_split = 0.2,
  callbacks = tensorboard_MLP_2
)
time.taken_PCA_MLP_2 <- proc.time() - startTime_PCA_MLP_2
plot(history_PCA_MLP_2)

# Evaluate the model
score_2 <- model_PCA_MLP_2 %>% evaluate(test.data, testLabels, batch_size = 128)
# Print the score
print(score_2)

# Predict the classes for the test data
classes2 <- model_PCA_MLP_2 %>% predict_classes(test.data, batch_size = 128)
# Confusion matrix
table(testTarget, classes2)
write.table(table(testTarget, classes2),file="Output\\confusionMatrix_Model_PCA_MLP_2.txt", sep="\t")


##### Model 3 RBF #####
# workingKeras   --> Full set (x+y) of training + testing dataset 
# For consistency, use the 4 variables below
# train.data_PCA --> "x" values for training dataset
# trainLabels    --> "y" values for training dataset
# test.data      --> "x" values for testing dataset
# testLabels     --> "y" values for testing dataset
trainRBFModel <- function(getNetworkSize, getNetworkIteration){
  setwd("/home/iss-user/Desktop/humandata")
  #install.packages('RSNNS')
  library(RSNNS)
  networkSize <- getNetworkSize
  networkIteration <- getNetworkIteration

  start.time <- Sys.time()
  rbfModel <- rbf(train.data_PCA, 
                  trainLabels, 
                  size= networkSize, 
                  maxit=networkIteration,
                  initFunc = "RBF_Weights",
                  learnFunc = "RadialBasisLearning",
                  initFuncParams=c(0, 1, 0, 0.01, 0.01),
                  learnFuncParams=c(1e-8, 0, 1e-8, 0.1, 0.8), 
                  shufflePatterns = TRUE,
                  linOut=TRUE)
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  
  #Testing Data
  #Prediction with test inputs
  rbf_predictions <- predict(rbfModel,test.data)
  
  #Generating Confusion Matrix
  getConfusionMatrix <- confusionMatrix(testLabels,rbf_predictions)
  
  #Training Data
  rbf_predictions_training <- predict(rbfModel,train.data_PCA)
  getConfusionMatrixTraining <- confusionMatrix(trainLabels,rbf_predictions_training)
  
  #Receiver operating characteristic (ROC) curve. (Accuracy)
  rPlotFilename <- paste(c('Rplot',networkSize,'-',networkIteration,'.png'), collapse='')
  png(filename=rPlotFilename)
  plotROC(rbf_predictions, testLabels)
  dev.off()
  
  #Loss Function 
  #Plot the iterative training and test error of the network
  sseFilename <- paste(c('SSE',networkSize,'-',networkIteration,'.png'), collapse='')
  png(filename=sseFilename)
  plotIterativeError(rbfModel)
  dev.off()
  
  regressionFilename <- paste(c('RegressionError',networkSize,'-',networkIteration,'.png'), collapse='')
  png(filename=regressionFilename)
  plotRegressionError(testLabels, rbf_predictions)
  dev.off()
  
  
  #Use Package ramify for argmax
  #install.packages('ramify')
  library(ramify)
  #Used to convert matrix to single column value
  model3Output <- argmax(rbf_predictions, rows=TRUE)
  
  #Others Stats
  weightMatrix(rbfModel)
  summary(rbfModel)
  info <- extractNetInfo(rbfModel)
  
  #Write Network Information to File "RBF-Info-{Size}-{Iteration}.txt"
  rbfInfo <- paste(c('RBF-Info-',networkSize,'-',networkIteration,'.txt'), collapse='')
  confusionMatrixInfo <- paste(c('ConfusionMatrix-',networkSize,'-',networkIteration,'.txt'), collapse='')
  
  time.taken.text <- paste(c('Time Taken: ',time.taken), collapse='')
  cat(capture.output(print(info), file=rbfInfo))
  
  cat('Testing Data Confusion Matrix(See Below)', file=confusionMatrixInfo, append=TRUE, sep = "\n")
  cat(capture.output(print(getConfusionMatrix), file=confusionMatrixInfo, append=TRUE))
  cat('Training Data Confusion Matrix(See Below)', file=confusionMatrixInfo, append=TRUE, sep = "\n")
  cat(capture.output(print(getConfusionMatrixTraining), file=confusionMatrixInfo, append=TRUE))
  cat(capture.output(print(time.taken), file=confusionMatrixInfo, append=TRUE))
  
  return (model3Output)
}

#Running various iterations for different nodes
trainRBFModel(100, 100)
trainRBFModel(500, 100)
trainRBFModel(1000, 100) #Optimal Model
trainRBFModel(1500, 100)
trainRBFModel(2000, 100)
trainRBFModel(2500, 100)

##### PRINT ALL THE SCORES #####
toWriteHeader <- c("Model no.", "Score Type", "Value", "User Time", "System Time", "Elapsed Time", " ", " ")
toWrite1L <- c("Model 1", "loss", score$loss, time.taken_PCA_MLP_1) 
toWrite1A <- c("Model 1", "accuracy", score$acc, time.taken_PCA_MLP_1)
toWrite2L <- c("Model 2", "loss", score_2$loss, time.taken_PCA_MLP_2) 
toWrite2A <- c("Model 2", "accuracy", score_2$acc, time.taken_PCA_MLP_2)
toWrite <- c(toWriteHeader, toWrite1L, toWrite1A, toWrite2L, toWrite2A)
write(toWrite, file = "Output\\Output_Score_Ensemble.txt", ncolumns = 8, sep = "\t")

##### OUTPUT FOR ENSEMBLE #####
forEnsemble <- cbind(classes, classes2)
forEnsemble <- forEnsemble + 1

#Function to compare emsemble of models. Input vector model1, model2, model3 parameter
getEnsembleResults <- function(model1, model2, model3){
  newEmsembleDF <- data.frame()
  if(length(model1) == length(model2) && length(model2) == length(model3)){
    model1Value <- 0
    model2Value <- 0
    model3Value <- 0
    for (i in 1:length(model1)) { 
      print(i)
      model1Value <- model1[i]
      model2Value <- model2[i]
      model3Value <- model3[i]
      newValue <- 0
      if(model1Value == model2Value){
        newValue <- c(model1Value)
        newEmsembleDF <- rbind(newEmsembleDF, newValue)
      }else if(model2Value == model3Value){
        newValue <- c(model2Value)
        newEmsembleDF <- rbind(newEmsembleDF, newValue)
      }else if(model1Value == model3Value){
        newValue <- c(model3Value)
        newEmsembleDF <- rbind(newEmsembleDF, newValue)
      }else{
        #Model 1 and Model 2 and Model 3 all have different results
        #Get the model with highest accuraccy
        newEmsembleDF <- rbind(newEmsembleDF, model1Value)
      }
      printToFile <- paste(c(model1Value,' ',model2Value,' ',model3Value,' ',newValue), collapse='')
      cat(capture.output(print(printToFile), file='ensemble-results.txt', append=TRUE))
    }
  }
  return (newEmsembleDF)
}

#Call function to generate the best model using Majority Voting method
model1Output <- classes[] + 1
model2Output <- classes2[] + 1
getEnsembleDF <- getEnsembleResults(model1Output, model2Output, model3Output)
getEnsembleDFExpanded <- decodeClassLabels(getEnsembleDF[,1])
ensembleConfusionMatrix <- confusionMatrix(testLabels,getEnsembleDFExpanded)

ensembleConfusionMatrix <- confusionMatrix(testLabels,getEnsembleDFExpanded)
cat(model1Confusion, file='model1-confusion-matrix.txt')
cat(capture.output(print(ensembleConfusionMatrix), file='ensemble-confusion-matrix.txt'))
