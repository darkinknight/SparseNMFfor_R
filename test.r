setwd ('E:/Project/Rcpptest')
library(Rcpp)
library(RcppArmadillo)
library(RcppEigen)
sourceCpp("spNMF.cpp")

A <- matrix(c(1,2,3,4,2,5,6,1,4),nrow = 3,byrow=TRUE)
b <- matrix(c(1,2,3),nrow = 3)
x0 <- matrix(c(0.1,0.1,0.1),nrow = 3)
y0 <- matrix(c(0.5,0.4,0.8),nrow = 3)

data_matrix = do.call(rbind,data)
##2维数据结构
T = Rcpp_m2NMF_fbs(data_matrix , 20, 0.01, 0.01)

##3列表数据结构
data2m = read.csv("ratings1.csv",header = FALSE)
data2m_matrix = do.call(cbind,data2m)
T = Rcpp_m3NMF_fbs(data2m_matrix, 20, 0.01, 0.01)
