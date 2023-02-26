library(lpSolve)

objective.fn <- c(900,2000)
const.mat <- matrix(
  c(1, 0, 
    0, 1,
    1, 1,
    0.2 ,1/3), ncol = 2, byrow = T)
const.dir <- c(">=", ">=", ">=", "<=")
const.rhs <- c(400, 200, 800, 168)

lp.solution <- lp("max", objective.fn, const.mat, const.dir, const.rhs, compute.sens = T)
lp.solution$solution
lp.solution$sens.coef.to
lp.solution$sens.coef.from
lp.solution$duals
