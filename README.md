# SparseNMFfor_R

$V = WH (W,H \in R_+)$

$f(x) = \frac{1}{2}||V - WH||^2_2 + \lambda_w|W|_1 + \lambda_h |H|_1$

方法基于ANSL + newton interior + fbs，主打一个适用稀疏数据

 Rcpp_m2NMF_fbs ： 为牛顿内点法的二维矩阵存储形式

 Rcpp_m3NMF_fbs ： 为牛顿内点法的三列表矩阵存储形式
 
 输入 data, k, lambda_w, lambda_h   (string method = "direct" , double tol = 1e-6, int maxiter = 10000,int thread = 5)
 
 输出model结构包括（W,H,loss）

 表现待证

```
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
```
