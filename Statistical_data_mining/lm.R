library(class)
library(MASS)
library(pROC)

#############################################################
#                     ROC FUNCTION                          #
#############################################################

myROC <- function(predictions, real){
threshold = seq(min(predictions) + 0.01, max(predictions) -0.01, by=0.01) 

sensitivityROC = c()
false_positiveROC = c()

for (val in threshold) {
  logistic_predic = rep("0", length(real)) ## prediccion 0: baja incidencia de fuego
  logistic_predic[predictions >= val] = "1"
  confusion = table(logistic_predic, real)
  sensitivity = confusion[2,2] / sum(confusion[ ,2]) ## proportion of Y=1 properly classified
  false_positive = confusion[2,1] / sum(confusion[  ,1]) ##
  sensitivityROC = c(sensitivityROC, sensitivity)
  false_positiveROC =  c(false_positiveROC, false_positive)
}

false_positiveROC = c(1,false_positiveROC,0)
sensitivityROC = c(1, sensitivityROC, 0)
points = cbind(false_positiveROC, sensitivityROC)

plot <- plot(false_positiveROC, sensitivityROC, type="l", main="ROC curve")
return(plot)
}



#############################################################
#############################################################

path <- "/home/jan/Downloads"

setwd(path)
data <- read.csv("incendios.csv", sep = ";")

index <- sample(1:nrow(data), nrow(data) * 0.8)

data_train <- data[index, ]
data_test <- data[-index, ]

summary(data_train)

# ---------------- GLM ----------------
glm1 <- glm(Y~., data = data_train, family = binomial)
pred_glm1_prob <- predict(glm1, data_test, type = "response")
pred_glm1 <- ifelse(pred_glm1_prob > 0.5, 1, 0)
conf_matrix_glm1 <- table(pred_glm1, data_test$Y)
summary(glm1)

acc_glm1 <- (conf_matrix_glm1[1, 1] + conf_matrix_glm1[2, 2]) / nrow(data_test)
myROC(pred_glm1_prob, data_test$Y)

auc_glm1 <- auc(data_test_signif_var$Y, as.numeric(pred_glm2_prob))
plot(roc(data_test_signif_var$Y, as.numeric(pred_glm2_prob)))

# The higest significant coefficientes are maquin_d, gan_for, prestur

data_train_signif_var <- data_train[, c(1,2,6,7)]
data_test_signif_var <- data_test[, c(1,2,6,7)]

glm2 <- glm(Y ~ ., data = data_train_signif_var, family = binomial)
pred_glm2_prob <- predict(glm1, data_test, type = "response")
pred_glm2 <- ifelse(pred_glm1_prob > 0.5, 1, 0)
conf_matrix_glm2 <- table(pred_glm2, data_test_signif_var$Y)

acc_glm2 <- (conf_matrix_glm2[1, 1] + conf_matrix_glm2[2, 2]) / nrow(data_test_signif_var)
myROC(pred_glm2_prob, data_test_signif_var$Y)

auc_glm2 <- auc(data_test_signif_var$Y, as.numeric(pred_glm2_prob))
plot(roc(data_test_signif_var$Y, as.numeric(pred_glm2_prob)))


# ---------------- LDA ----------------
model_lda <- lda(formula = Y ~ ., data = data_train_signif_var)
pred_lda_prob <- predict(model_lda, data_test_signif_var)$posterior[, 2]
pred_lda <- predict(model_lda, data_test_signif_var)$class
conf_matrix_lda <- table(pred_lda, data_test_signif_var$Y)

acc_lda <- (conf_matrix_lda[1, 1] + conf_matrix_lda[2, 2]) / nrow(data_test_signif_var)
myROC(pred_lda_prob, data_test_signif_var$Y)

auc_lda <- auc(data_test_signif_var$Y, as.numeric(pred_lda_prob))
plot(roc(data_test_signif_var$Y, as.numeric(pred_lda_prob)))

# ---------------- kNN ----------------
k_choose <- function(data_train, data_test){
  err <- 100000
  k_fit <- 100000
  last <- ncol(data_train)
  for(k in 1:20){
    model_temp <- knn(data_train[, -last], data_test[, -last], cl = data_train[, last], k = k)
    err_temp <- mean(model_temp == data_test[, last])  #classification error
    if(err_temp < err){
      err <- err_temp
      k_fit <- k
    }
  }
  return(k)
}

k <- k_choose(data_train_signif_var, data_test_signif_var)
data_train_signif_var[, 1:3] <- scale(data_train_signif_var[, 1:3])
data_test_signif_var[, 1:3] <- scale(data_test_signif_var[, 1:3])

model_knn <- knn(data_train_signif_var[, -4], data_test_signif_var[, -4], cl = data_train_signif_var[, 4], k = k)
model_knn_prob <- knn(data_train_signif_var[, -4], data_test_signif_var[, -4], cl = data_train_signif_var[, 4], k = k, prob = T)
conf_matrix_knn <- table(model_knn, data_test_signif_var$Y)

acc_knn <- (conf_matrix_knn[1, 1] + conf_matrix_knn[2, 2]) / nrow(data_test_signif_var)
myROC(as.numeric(model_knn_prob), data_test_signif_var$Y)

auc_knn <- auc(data_test_signif_var$Y, as.numeric(model_knn_prob))
plot(roc(data_test_signif_var$Y, as.numeric(model_knn_prob)))

summary <- matrix(c(auc_glm1, auc_glm2, auc_lda, auc_knn, acc_glm1, acc_glm2, acc_lda, acc_knn), ncol = 4)
colnames(summary) <- c("glm1", "glm2", "lda", "knn")
rownames(summary) <- c("auc", "accurency")
summary

# Results:
# We observe that auc grows when we remove no significant coefficientes and accurency is lower in glm models.
# It is resonable because more predictors always rises accurency. I would choose second glm model because of auc and
# is easier to interpretaction. LDA model is doing well but assumptions (normal distribution of predictors) 
# arent satisfied (binary variable). kNN looks good. I wrote function which choose k that minimalize classification error
# but results are worst than glm2. I scaled data before knn because this algorithm bases on distance metric.
# After every model I print ROC curves optained in two diferent ways - first from function written in class, 
# second from pROC package
