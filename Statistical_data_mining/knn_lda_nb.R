#          Jan Wasilewski (Erasmus)  
# Minería Estadística de Datos (Máster en Matemáticas)
# Trabajo sobre los contenidos del Tema 1
# data source : https://archive.ics.uci.edu/ml/datasets/adult


library(tidyverse) # plots, data transformations
library(plyr) # data transformations
library(reshape2) # plots
library(e1071) #naive bayes
library(caret) # kNN
library(MASS) # lda
library(ROCR) # ROC curve

# ------------- Preprocesamiento --------------
path <- "/home/jan/Documents/Uni/Data mining/adult_dataset.csv"

data <- read_csv(path, col_names = F)
colnames <- c("age", "workclass", "fnlwgt", "education",
                       "education_num", "marital_status", "occupation",
                       "relationship", "race", "sex", "capital_gain", "capital_loss",
                       "hours_per_week", "native_country", "income")

colnames(data) <- colnames

data %>% summary
apply(data, 2, class)

continuous_index <- colnames[c(1, 3, 5, 11:13)] 
categorical_index <- colnames[which(!(1:15 %in% c(1, 3, 5, 11:13)))]

data[continuous_index] <- data[continuous_index] %>% lapply(as.integer)  
data[categorical_index] <- data[categorical_index] %>% lapply(factor)  
data$income <- data$income %>% revalue(c("<=50K" = 0, ">50K" = 1)) %>% factor
sapply(data, class)

data %>% summary
index <- sample(1:nrow(data), round(.7 * nrow(data)))
data_train <- data[index, ]
data_test <- data[-index, ]

# -------------- Visualización -----------------
data_plot_categorical <- melt(data_train[categorical_index], "income")
data_plot_categorical %>% ggplot(aes(x = value, fill = income)) +
  geom_bar() + 
  facet_wrap(~ variable, ncol = 2, scales = "free") +
  theme_minimal()

data_plot_continuous <- melt(data_train[c(continuous_index, "income")], "income")
data_plot_continuous %>% ggplot(aes(x = value, fill = income)) + 
  stat_density(position = "identity", alpha = .5) + 
  facet_wrap(~ variable, ncol = 2, scales = "free") +
  theme_minimal()

# -------------- Function de Resultados --------------
Results_function <- function(predict_prob, real_values = data_test$income, treshold = .5) {
  predict <- cut(predict_prob$value, c(0, treshold, 1), labels = 0:1, include.lowest = T)
  confusion_matrix <- table(predict, real_values)
  Accuracy <- (confusion_matrix[1, 1] + confusion_matrix[2, 2]) / length(data_test$income)
  Sensitivity <- confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[2, 1])
  Specificity <- confusion_matrix[2, 2] / (confusion_matrix[1, 2] + confusion_matrix[2, 2])
  PPV <- confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 2])
  NPV <- confusion_matrix[2, 2] / (confusion_matrix[2, 1] + confusion_matrix[2, 2])
  pred_ROC <- prediction(predict_prob$value, real_values)
  perf_ROC <- performance(pred_ROC, "tpr", "fpr")
  plot(perf_ROC, colorize = F)
  abline(0,1)
  title(paste0("ROC curve for ", predict_prob$name))
  perf_AUC <- performance(pred_ROC, measure = "auc")
  AUC <- perf_AUC@y.values[[1]]

  return(list(confusion_matrix = confusion_matrix,
              Accuracy = Accuracy,
              Sensitivity = Sensitivity,
              Specificity = Specificity,
              PPV = PPV, NPV = NPV, AUC = AUC))
}

# ---------------- Modelos ----------------------
model_naiveBayes <- naiveBayes(income~., data = data_train)
pred_naiveBayes <- list(value = predict(model_naiveBayes, data_test, type = "raw")[, 2], name = "Naive Bayes")

model_lda <- lda(income ~., data = data_train)
pred_lda <- list(value = predict(model_lda, data_test)$posterior[, 2], name = "LDA")

# knn - Ajuste de hiperparámetros (es necesaria la normalización de datos!)
Accuracy_knn <- c()
# la atención requirió muchos cálculos
for(i in 1:20) {
  model_knn_temp <- knn3(scale(data_train[continuous_index]), data_train$income, k = i)
  pred_knn_temp <- predict(model_knn_temp, scale(data_test[continuous_index]))[, 2]
  pred_knn_temp_class <- cut(pred_knn_temp, c(0, .5, 1), labels = 1:2, include.lowest = T)
  conf_matrix <- table(pred_knn_temp_class, data_test$income)
  Accuracy_knn_temp <- (conf_matrix[1, 1] + conf_matrix[1, 2]) / length(data_test$income)
  Accuracy_knn <- c(Accuracy_knn, Accuracy_knn_temp)
  }

plot(1:20, Accuracy_knn, type = "b")
abline(v = 4, col = "blue")
title("Elbow plot for accurancy kNN")

#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
# Otra metoda conoser k (con validación cruzada).
index_knn <- sample(1:nrow(data_train), round(0.6 * nrow(data_train)))
data_train_knn_x <- data_train[index_knn, -15] %>% as.data.frame
data_train_knn_y <- data_train[index_knn, 15]
data_val_knn_x <- data_train[-index_knn, -15] %>% as.data.frame
data_val_knn_y <- data_train[-index_knn, 15] 

plot_help <- c()
for(k in 1:40){
  model_knn_temp <- knn3(scale(data_train_knn_x[continuous_index]), unlist(data_train_knn_y), k = k)
  pred_knn_temp <- predict(model_knn_temp, scale(data_val_knn_x[continuous_index]))[, 2]
  pred_knn_temp_class <- cut(pred_knn_temp, c(0, .5, 1), labels = 0:1, include.lowest = T)
  plot_help <- c(plot_help, mean(pred_knn_temp_class != unlist(data_val_knn_y)))
}


plot(1:40, plot_help, type = "l")
abline(v = which.min(plot_help))
which.min(plot_help)
# De este metoda eligió numero que error es el mas peqenio

#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------


model_knn_temp <- knn3(scale(data_train[continuous_index]), data_train$income, k = which.min(plot_help))
pred_knn <- list(value = predict(model_knn_temp, scale(data_test[continuous_index]))[, 2], name = "knn")

# ---------------- Results ---------------------
sapply(list(naiveBayes = pred_naiveBayes, lda = pred_lda, knn = pred_knn), Results_function)

#              Comparación de resultados
# Podemos ver que el modelo LDA funciona mejor (AUC, precisión). 
# También es visible en curvas ROC. El comportamiento extraño de
# kNN (diagrama de codo oscilante) puede explicarse por un número
# impar de observaciones. Sin embargo, los requisitos no se suturan
# (normalidad de la distribución, correlacion de variables - esto debe ser atendido).
# También se recomienda cuidar los valores atípicos.Para la predicción, 
# elegiría el modelo LDA.
