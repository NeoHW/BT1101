library(TTR)
library(tsbox)
library(lpSolve)

week <- c(1:10)
pack <- c(13,17,32,33,31,33,36,36,37,39)
table1 = cbind(week,pack) 
ts_ts(table1)
for (k in 1:10){
  pack.sma <- SMA(pack , n = k)
  print(k)
  print(sqrt(mean((pack-pack.sma)^2 , na.rm = T)))
}

pack.sma <- SMA(pack , n = 2)
pack.sma

objective.fn <- c(1,1,1)
const.mat <- matrix(c(
    # Camera constraints
    3 , 2 , 1,
    # LED constraints
    0 , 2 , 4,
    # Motor constraints
    2 , 1 , 1,
    # Battery constraints
    2 , 2 , 2
    ), ncol = 3, byrow = T)
const.dir <- c("<=", "<=", "<=" , "<=")
const.rhs <- c(200, 400, 900 , 252)

lp.solution <- lp("max", objective.fn, const.mat, const.dir, const.rhs, compute.sens = T , int.vec = c(1,2,3))
lp.solution$solution
lp.solution
