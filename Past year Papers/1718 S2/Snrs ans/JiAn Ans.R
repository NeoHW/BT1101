library(lpSolve)
library(TTR)
library(tsbox)

objective.fn <- c(0.48, 0.39, 0.21,0.23,0.15,0.11,0.66,0.25,0.52)
const.mat <- matrix(
  c(15.7, 13.1, 8.6,17.2,6.7,12.4,17.4,13.9,15.3, 
    28.8, 4.3, 4.7,6.2,3.9,1.5,14.7,4.6,12.4,
    25.7, 5.6, 2.8,3.1,1.7,2.5,23.4,13.2,15.3), ncol = 9, byrow = T)
const.dir <- c(">=", ">=", "<=")
const.rhs <- c(11, 28, 4)

lp.solution <- lp("min", objective.fn, const.mat, const.dir, const.rhs, compute.sens = T)
lp.solution$solution
lp.solution

objective.fn2 <- c(12.4, 13.75, 12.30,12.15,18.35,
                   8.5,18.55,8.8,9.85,16.45,
                   9.3,14.25,6.45,11.15,14.95)
const.mat2 <- matrix(c(
    # Constraint 1: capacity constraint
    rep(0,0) , rep(1,5) , rep(0,10),
    rep(0,5) , rep(1,5) , rep(0,5),
    rep(0,10), rep(1,5) , rep(0,0),
    # Constraint 2: demand constraint
    rep(c(1,0,0,0,0) , 3) , 
    rep(c(0,1,0,0,0) , 3) ,
    rep(c(0,0,1,0,0) , 3) ,
    rep(c(0,0,0,1,0) , 3) , 
    rep(c(0,0,0,0,1) , 3)
    ), ncol = 15, byrow = T)
const.dir2 <- c(rep("<=",3), rep(">=",5))
const.rhs2 <- c(1300,900,400,160,330,600,590,880)

lp.solution2 <- lp("min", objective.fn2, const.mat2, const.dir2, const.rhs2 , int.vec = c(1:15))
matrix(c(lp.solution2$solution), ncol = 5, byrow = T)
lp.solution2


objective.fn3 <- c(12.4, 13.75, 12.30,12.15,18.35,
                   8.5,18.55,8.8,9.85,16.45,
                   9.3,14.25,6.45,11.15,14.95,
                   11.4,10.6,9.75,13.55,11.95,
                   14.1,15.5,13.85,9.45,12.25,
                   0,0,0,0,0)
const.mat3 <- matrix(c(
  # Constraint 1: capacity constraint
  rep(0,0) , rep(1,5) , rep(0,25),
  rep(0,5) , rep(1,5) , rep(0,20),
  rep(0,10), rep(1,5) , rep(0,15),
  rep(0,15), rep(1,5) , rep(0,10),
  rep(0,20), rep(1,5) , rep(0,5),
  # Constraint 2: demand constraint
  rep(c(1,0,0,0,0) , 5) , rep(0,5),
  rep(c(0,1,0,0,0) , 5) , rep(0,5),
  rep(c(0,0,1,0,0) , 5) , rep(0,5),
  rep(c(0,0,0,1,0) , 5) , rep(0,5), 
  rep(c(0,0,0,0,1) , 5) , rep(0,5),
  # Constraint 3: either factory D or E
  rep(0,25) , 1 , 1 , 1 , 0 , 0,
  rep(0,28) , 1 , 1 
  
), ncol = 30, byrow = T)
const.dir3 <- c(rep("<=",5), rep(">=",5) , "<=" ,"<=")
const.rhs3 <- c(1300,900,400,1200,1200,160*1.1,330*1.1,600*1.1,590*1.1,880*1.1,3,1)

lp.solution3 <- lp("min", objective.fn3, const.mat3, const.dir3, const.rhs3 , int.vec = c(1:25) , binary.vec = c(26:30))
matrix(c(lp.solution3$solution), ncol = 6, byrow = T)
lp.solution3

week <- c(1:10)
tins <- c(15,18,37,40,39,38,35,41,55,30)
table1 = cbind(week,tins) 
ts_ts(table1)
for (k in 1:10){
  tins.sma <- SMA(tins , n = k)
  print(k)
  print(mean((tins-tins.sma)^2 , na.rm = T))
}
tins.sma <- SMA(tins , n = 2)








