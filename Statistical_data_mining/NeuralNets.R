library(tidyverse)
library(nnet)

# Parte 1
data <- tibble(x1 = rnorm(4000), x2 = rexp(4000), y = x1^2 + x2 + rnorm(4000, 0 , 0.1))


data <- data %>% mutate_at(1:2, function(x) (x - mean(x)) / sd(x)) %>% mutate_at(3, function(x) (x-min(x))/(max(x)-min(x)))

index <- sample(1:4000, 3000)

data_train_x <- data %>% select(-y) %>% slice(index)
data_train_y <- data %>% select(y) %>% slice(index)
data_test_x <- data %>% select(-y) %>% slice(-index)
data_test_y <- data %>% select(y) %>% slice(-index)

nn <- nnet(data_train_x, data_train_y, size = 1)
pred <- predict(nn, data_test_x, type = "raw")
MSE <- mean(unlist((pred-data_test_y)^2))

predict(nn, c(1,1), type = "raw")

nn$wts
nn$n

# Parte 2

setwd("/home/jan/Documents/Uni/Data mining")

set.seed(0409)

data_train <- read_csv("mnist_train.csv", col_names = F)
data_test <- read_csv("mnist_test.csv", col_names = F)

data_train %>% summary
data_train %>% dim
data_train %>% select(785) %>% summary
data_test %>% summary
data_test %>% dim

data_train_norm <- data_train %>% mutate_at(1:784, function(x) x/255)
data_test_norm <- data_test %>% mutate_at(1:784, function(x) x/255)

numbers <- sample(0:9, 5)

train <- data_train_norm %>% filter(X785 %in% numbers) %>% slice(sample(1:n(), 3000))
test <- data_test_norm %>% filter(X785 %in% numbers) %>% slice(sample(1:n(), 1000))

layout(matrix(1:25, 5))
train %>% slice(1:25) %>%
          select(-785) %>%
          mutate_all(function(x) round(x*255)) %>%
          as.matrix %>%
          apply(1, function(x) image(t(apply(matrix(x, 28, byrow = FALSE), 1, rev)),
                                     col = hcl.colors(12, "Grays", rev = TRUE)))

test %>% slice(1:25) %>%
  select(-785) %>%
  mutate_all(function(x) round(x*255)) %>%
  as.matrix %>%
  apply(1, function(x) image(t(apply(matrix(x, 28, byrow = FALSE), 1, rev)),
                             col = hcl.colors(12, "Grays", rev = TRUE)))

index <- test %>% select(-785) %>% apply(2, function(x) if(min(x) == max(x)) T else F) %>%  which

train_pc_x <- train %>% select(-c(index, 785)) %>% prcomp()
train_pc_y <- train %>% select(785)
test_pc_x <- test %>% select(-c(index, 785)) %>%  prcomp()
test_pc_y <- test %>% select(785)

nn <- nnet(X785 ~ ., size = 15, data =  train2)
