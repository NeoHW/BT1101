# BASICS

## uncategorised stuff
y <- as.character(3) # forcing y to have character class
p <- "pigs"
paste(y,p) # concatenates the 2 char values to print "3 pigs"
for(i in 1:upper_bound_inclusive) {code} # iterationnnn
seq(from, to, by=) # creating a vector of values
sample(vector_selected_from, size=20, replace=FALSE) # randomly select 20 values from a vector without replacement

matrix(1:9, byrow = TRUE, nrow = 3) # first argument is collection of elements to be arranged into the rows and columns of the matrix; second argument byrow indicates that the matrix is filled by the rows; third argument nrow indicates that matrix should have three rows
cbind(matrix1, matrix2, vector1) # binds matrices and/or vectors as new columns; rbind() for rows
matrix[1:3,2:4] # gives elements at row 1 to 3 and col 2 to 4

data.frame(vector1, vector2, vector3) # combines the vectors into a dataframe, where each vector represents 1 column worth of data
subset(data_frame, subset = condition) # picks out the rows that fulfill the boolean condition (can be targeted at one of the variables in the dataframe)

## dplyr 
select(data) # chooses data to print out; has helpers such as contains("variable_name"), starts_with("prefix"), ends_with() or last_col() [takes last column]
select(data, -variable) # removes a certain col corresponding to d variable
select(new_name = variable1, variable2, variable3) # renames variable; can do rename(new_col_name = old_col_name) instead also
mutate(data, manipulation_of_existing_variable) # adds new col to data 
mutate(data, new_variable = case_when(condition ~ value_to_assign, next_condition ~ next_value)) # assign values of new variable conditionally
transmute(variable1, variable2, new_name = variable3 / variable4) # works like a combination of select and mutate

count(data, variable, sort=T) # counts number of data entries in each category of the variable, sort from most to least common
count(data, state) # counts how many rows of data for each state there are
count(data, state, wt = population) # adds the populations of the rows of data belonging to the same state together; wt represents weight

summarize(data, new_variable = fn(existing_variable)) # create >= 1 new variables (separated by ",") using functions applied on existing variables; functions include sum(), mean(), median(), min(), max(), n() [size of group]
group_by(data, variable1, variable2) # groups data by variables; can subsequently summarise data by variables as well, eg %>% summarise(Freq=n()); effects reversed by ungroup()
arrange(data, variable) # arranges data in ascending order based on the variable given (use desc(variable) to arrange in descending order instead
filter(data, condition) # can put multiple conditions separated by ","
top_n(data, n, variable) # gives top n rows for the variable

## stats stuffs
dnorm(0, mean=0, sd=1)	# pdf, probability density
pnorm(2, mean=0, sd=1, lower.tail = TRUE)		# cumulative distribution function,  e.g. (find probability of DVD sales < 25)  # lower tail = T/F depends on H1
qnorm(0.90, mean=0, sd=1, lower.tail = TRUE)	# find 90th percentile of Z
rnorm(10, mean=0, sd=1)	# generate 10 standard normal random numbers

dt(x, df)	# probability density; x is vector of quantiles
pt(q, df, lower.tail = TRUE)	# cdf, distribution; q is vector of quantiles 
qt(p, df, lower.tail = TRUE)	# percentile of t; p is vector of probabilities
rt(n, df)	# generates random numbers from t; n is number of observations, if length(n) > 1, length is taken to be the number required

## reading in data from excel
setwd('/Users/haowei/NUS/BT1101/Week 4')
library("readxl")
bank_credit_risk_data <- read_excel('Bank Credit Risk Data.xlsx', sheet = "Base Data", skip = 2)
BD <- bank_credit_risk_data

## Reading in from CSV
d1 <- read.csv("CardioGoodFitness.csv")

## Getting summary of the dataset
summary(dataset)

## Removing scientific Notation
options(scipen = 999) # default = 0

# LECT & TUT 3 : Data visualisation, Tabulation & Frequencies

## creating some tables (contingency table under barplot)
### frequency table (2 cols, one for gender and one for n, which is frequency)
genderFreq = data %>% count(Gender)
kable (genderFreq, caption = "Frequency of Data by Gender") 

### pivot table
rpivotTable(BD, rows ="work_year", cols=c("experience_level","employment_type"),aggregatorName = "Count")

## pareto analysis 
BD.sav <- BD %>% select(Savings) %>% arrange(desc(Savings))
BD.sav$Percentage <- BD.sav$Savings / sum(BD.sav$Savings) # compute % of savings over total savings
BD.sav$Cumulative <- cumsum(BD.sav$Percentage) # compute cumulative % for Savings
BD.sav$Cumulative.cust <- as.numeric(rownames(BD))/nrow(BD) # generates many % in increasing order to show % of cumulative savings over total savings with each customer included 1 by 1 # kinda useless step
which(BD.sav$Cumulative>0.8)[1] # computes the number of customers contributing to 80% of the total savings
which(BD.sav$Cumulative>0.8)[1]/nrow(BD) # compute % of customers with top 80% savings
# conclusion: From the Pareto Analysis, it is evident that around 146 (out of 722) customers that carried out transactions with the highest amounts made up 80% of the total transaction amounts. This number is around 20% of the customers in the bank. Hence, there is indeed a small proportion of transactions that contributed the most to total amount ($) in bank account transactions in the month. It is a 80-20 split.
quantile(data, c(percentile1, percentile2)) # can calc some percentiles here if needed (OPTIONAL)

## nice 7 rainbow colours to use for all d charts below
nice_colours = c("indianred1","goldenrod1","khaki1","seagreen1","cadetblue1","thistle1","lightsalmon")

## plotting multiple charts and graphs side by side
par(mfcol=c(2,3)) # before typing plotting functions results in charts appearing in 2 rows 3 cols

## pie chart function
piechartfunc <- function(vfreq, vlab, vcol, vtitle){
  vslice <- vfreq$n
  vpercent <- 100*round(vfreq$n/sum(vfreq$n),2)
  label<-vlab 
  label<-paste(label,",",sep="")
  label<-paste(label,vpercent) #default of sep=" "
  label<-paste(label,"%",sep="")
  pie(vslice,labels=label, col=vcol,radius=1, main=vtitle)
}
piechartfunc(HousingFreq, HousingFreq$Housing, c("lightgoldenrod1","cornflowerblue","lightcoral"), "Customer Housing")

## plotting histogram & frequency table using histogram breaks
# add extra prob=T to get hist of relative frequency (density) instead; manually add breaks using breaks= <number of cells OR vector giving breakpoints OR function to compute vector, eg seq(from, to, by)>
emp.hist <- hist(BD$`Months Employed`, main="Histogram of Customers' Months of Employment", xlab="Months Employed", ylab="No. of Customers", col=c("lightsalmon"), xlim=c(0,120), ylim=c(0,160), labels=TRUE) 
# Frequency table from histogram generated above
emp.group <- cut(BD$`Months Employed`, emp.hist$breaks) # add argument dig.lab = 6 to remove scientific notation
emp.table <- table(emp.group)
kable(emp.table, caption = "Distribution of Bank Customers by Months Employed")

## scatter & line plots whoop
### scatter with line lel
plot(x_axis_data, y_axis_data, main="Scatterplot of Impt Stuff", xlab="xxx", ylab="yyy", pch=18) 
lines(x_axis_data, y_axis_data) # connects the dots with straight lines

### lonely lines
plot(data_v,type="o",col="red", xlab="Month", ylab="Rainfall", main="Rainfall Chart")
lines(data_t, type="o", col="blue") # adds line to same plot
lines(data_u, type="o", col="green")
legend(x, y, legend=c("region v", "region t", "region u"), col=c("red", "blue", "green"), lty=1, cex=0.8) # x and y refer to coordinates to put legend at

## barplot 
### level 1 noob plot
par(mar=c(5,7,4,1)+.1) #(bottom, left, top, and right) #use to change plot margins in case labels get too long 
loanhrBP = barplot(LoanFreq$n, names.arg=LoanFreq$`Loan Purpose`, col="blue", main="Frequency of Loan Purpose", cex.names = 0.8, xlim=c(0,120), xlab="No. of Loans", horiz=TRUE, las=1) # horiz is by default false, determines if bars are long or tall; las = 1 or 2 determines if labels read horizontally or rotated 90 deg vertically
text(x = LoanHRFreq$n, y = loanhrBP, col = "Black", LoanHRFreq$n, cex = 0.8, pos = 4) # adds text label at the end of the bars, showing LoanHRFreq$n
mtext("Job Title", side = 2, line = 5, cex.lab = 1, las = 3) # use this instead of ylab so label will not overlap names , # line = 5 : label will be on line 5

### Contingency Table 
## or refer to T4 HTML
BDCRJ <- BD %>% group_by(Job, `Credit Risk`) %>% count() # remember to ungroup() if necessary
BDCRJ.spread <- BDCRJ %>% spread(key=Job, value=n)
BDCRJ.spread[is.na(BDCRJ.spread)] <- 0 
kable(BDCRJ.spread, caption = "Contingency table for Job & Credit Risk") # resultant table has leftmost col Credit Risk (with d categories below), and subsequent cols are the diff Job categories (and their respective freq below)

### Grouped Barplots
CRJbarmatrix <- as.matrix(BDCRJ.spread[,2:5]) # extract the data from the different job types in a matrix - this will be d diff groups
CRJbar_col <- c("lightgoldenrod1", "lightcoral")
bpjob <- barplot(CRJbarmatrix, col=CRJbar_col, ylim=c(0,300), main="Job and Credit Risk", ylab="No. of Customers", xlab="Job", beside = TRUE) # gives different groups of barplots, diff bars from each group are the diff categories from the leftmost col excluded from matrix (ie Credit Risk in this case); by default the bars from each group are stacked but can change to stick together side by side using `beside=T`)
legend("topright", cex=1, fill=CRJbar_col, legend=BDCRJ.spread$`Credit Risk`) # last input is the labels for the diff bars in each group ie the leftmost/leftout col
text(bpjob, 0, col = "black", CRJbarmatrix, cex=1, pos=3) #creating numbers in the barplot

### boss barplots part 2 after changing categorical data in numerical form to factor data
titanic_table <- titanic_train %>% group_by(Survived, Sex) %>% summarize(number=n()) %>% mutate(Survived = factor(Survived, levels=c(0,1), labels=c("No", "Yes")))

ggplot(titanic_table, aes(x=Sex, y=number, fill=Survived)) + geom_bar(stat = "identity") + theme_bw()



# LECT & TUT 4 : Statstical Measures, Probability Distributions & Data modelling, outlier analysis

## getting descriptive stats
### using describe(By) function from psych package
dfage <- describe(BD$Age, IQR=TRUE) # stats include count (n), mean, sd, median, min, max, range, skew, kurtosis (calcuated value is less by 3 so degree of dispersion determined by CK > 0 or < 0 instead of 3), IQR
dfsavings <- describe(BD$Savings, IQR=TRUE)
df.desc1 <- rbind(dfage,dfsavings) # bind so table later will have stats for both variables
df.desc1$trimmed <- df.desc1$mad <- df.desc1$se <- NULL # remove se, mad and trimmed if not needed
df.desc1$vars[1]<-"Age" # naming the row
df.desc1$vars[2]<-"Savings"
kable(df.desc1, row.names = FALSE, caption = "Descriptive Statistics for Age and Savings")

mat.ALP <- describeBy(BD$Age, group=BD$`Loan Purpose`, mat=TRUE, IQR=TRUE) # grouped by single variable
mat.ALP <- mat.ALP[,-c(1,3,7,8,15)] # remove item, vars, trimmed, mad and se columns in the matrix
kable(mat.ALP, caption = "Descriptive Statistics for Age grouped by Loan Purpose", row.names = FALSE)

mat.ALP2 <- describeBy(BD$Age, group=list(BD$`Loan Purpose`, BD$Gender), mat=TRUE, IQR=TRUE) # grouped by > 1 variable
mat.ALP2 <- mat.ALP2[,-c(1,4,9,10,16)] # remove item, vars, trimmed, mad and se columns in the matrix
kable(mat.ALP2, caption = "Descriptive Statistics for Age grouped by Loan Purpose & Gender", row.names = FALSE)

### noob manual calc of the stuff describe() can do
dsAmount <- DS %>% summarise(vars='Salary in USD',n=n(), mean=mean(salary_in_usd), sd=sd(salary_in_usd), median=median(salary_in_usd), skew=skew(salary_in_usd), kurtosis=kurtosi(salary_in_usd))
kable(dsAmount, row.names = FALSE, caption = "Description Statistics for Salary in USD - manual", digits=3) # digits specify number of dp for non integer numbers

### mode
names(table(data)[table(data) == max(table(data))])

### test for correlation
corr.test(x = x.var, y = y.var) # check p-value, if p-value < 0.05, insufficient evidence to reject null hypothesis so data is correlated

### covariance and correlation coefficient
covariance.AS <- cov(BD$Age, BD$Savings) # +ve/-ve -> direct/inverse relationship
correlation.AS <- cor(BD$Age, BD$Savings) # +ve/-ve -> direct/inverse relationship;     #0 = no relationship, <0.3 = weak linear relationship, 0.3-0.7 = moderate linear relationship, >0,7 = strong linear relationship; 
cts <- corr.test(df9[3:7]) # use corr.test() in psych package for multiple variables; 

## presenting the stats
### table for means of diff categories of a variable
mean.age <- BD %>% group_by(`Loan Purpose`) %>% summarise(mean=mean(Age)) 
kable(mean.age) # gives table of mean for each loan purpose

### nice barplot for d means 
mean.age.spread <- mean.age %>% spread(key=`Loan Purpose`,value=mean)
mat.meanage<-as.matrix(mean.age.spread[,c(1:10)])
par(mar=c(5,10,4,2)) # default plot margin is (5,4,4,2), this adds bigger left margin for the barplot
barplot(mat.meanage, horiz=TRUE, col =c("pink"), main="Mean Age across Loan Purpose Types", cex.names=0.9, las=1, xlim=c(0,50), xlab="Mean Age") 

## Outlier Analysis
### Check Historgram then
### Check for normality
shapiro.test(BD$Savings) # p-value < 0.05 means data not normally distributed; W < 0.5 means distribution significantly deviates from theoretical normal (w ranges from 0 to 1)
### boxplot (Not appropriate for outlier analysis when data is NOT NORMAL; use histogram instead if not normal)
onePointFiveIQROutliers <- boxplot(salarySEFT$salary_in_usd, main='Boxplot with range = 1.5', horizontal =  TRUE)
threeIQROutliers <- boxplot(salarySEFT$salary_in_usd, range=3, main='Boxplot with range = 3', horizontal =  TRUE)
mildOutliers <- setdiff(onePointFiveIQROutliers$out,threeIQROutliers$out) # Getting mild outliers  # setdiff(a,b) identifies values that are in a but not in b
extremeOutliers <- threeIQROutliers$out # Getting extreme outliers
mildOutliers
extremeOutliers

## dealing with unknown data distribution
### check for normality/normal distribution (shapiro, density plot, q-q plot; good to use visual and statistical technique to check, esp when one method alone not v conclusive)
shapiro.test(ST$Amount) # p-value < 0.05 means data not normally distributed; W < 0.5 means distribution significantly deviates from theoretical normal (w ranges from 0 to 1)
plot(density(ST$Amount)) # gives line graph, smoothed-out version of histogram

qqnorm(ST$Amount) # draws the correlation between a given sample and the normal distribution
qqline(ST$Amount, col=2) # plots 45-degree reference line

### if data not normal, can impute outliers with mean 
### (not learnt in 2022?)
D$Demand.imp <- D$Demand
D$Demand.imp[D$Demand>=1500]<-mean(D$Demand)

## probably calculating some probabilities...
### using raw data
d1 <- D %>% filter(Demand.imp > 800)
pr1 <- nrow(d1)/nrow(D) # gives probability of Demand > 800

### using mean and sd of normally distributed data to find probability for a single new observation
m <- mean(D$Demand.imp)
s <- sd(D$Demand.imp)
pr11 <- pnorm(800, mean=m, sd=s, lower.tail = FALSE) # lower.tail=F means P(Z>=800), and T means P(Z<=800)

#### if asked to find probability for MEAN of new n observations:
#### e.g. mean = 36, SD = 8; find probability that mean purchase amount for 16 customers > $40
standard_error <- 8 / sqrt(16)
1 - pnorm(40, 36, standard_error)


# LECT & TUT 5 : Sampling, Estimation, Hypo testing

## 95% confidence interval 
### confidence interval for mean
ci.age <- CI(BD$Age, ci=0.95) # gives upper, mean and lower
print(cbind(ci.age[3],ci.age[1]), digits=4)
# conclusion: The 95% CI for mean age is [33.34, 35.45]. We are 95% "confident" that the interval [33.34, 35.45] contains the true population mean; that with repeated sampling, there is 95% probability that the interval correctly estimates the true population mean.

### manual calc for CI for mean , assuming unknown population sd (t-distribution)
uCIage95t <- mean(BD$Age) - qt(0.025, df=nrow(BD)-1)*sd(BD$Age)/sqrt(nrow(BD)) # df = number of sample values - number of estimate parameters
lCIage95t <- mean(BD$Age) + qt(0.025, df=nrow(BD)-1)*sd(BD$Age)/sqrt(nrow(BD))
print(cbind(lCIage95t, uCIage95t), digits=4)

### manual calc for CI for proportion, assumes unknown population sd (z-distribution)
n.bd = nrow(BD)
age50 <- BD %>% filter(Age>50)
p50 = nrow(age50)/nrow(BD)
lCIp50 <- p50 + (qnorm(0.025)*sqrt(p50*(1-p50)/n.bd))
uCIp50 <- p50 - (qnorm(0.025)*sqrt(p50*(1-p50)/n.bd))
print(cbind(lCIp50, uCIp50), digits=3)
# conclusion: The 95% CI for proportion of Age to be greater than 50 is [0.0766, 0.135]. With repeated sampling, there is 95% probability that this interval will correctly estimate the true proportion of customers with age greater than 50.


## prediction intervals (! need check for normality of data first bc CLT cannot be applied !)
### checking for normality
qqnorm(BD$Age,ylab='Sample quartiles for Age')
qqline(BD$Age,col='red')
shapiro.test(BD$Age)

### if data is normal
mnage <- mean(BD$Age)
sdage <- sd(BD$Age)
n.bd <- nrow(BD)
uPI.age <- mnage - (qt(0.025, df = (n.bd-1))*sdage*sqrt(1+1/n.bd)) # calculating for 95% interval     # 0.05/2 = 0.025 in the formula
lPI.age <- mnage + (qt(0.025, df = (n.bd-1))*sdage*sqrt(1+1/n.bd))
cbind(lPI.age, uPI.age)
# conclusion: The 95% prediction interval for sales amount of a new DVD transaction is [14.49,25.63]. Given the observed DVD prices, the price of a new DVD will lie within this interval with a 95% level of confidence. With repeated sampling, 95% of such constructed predictive intervals would contain the new DVD price.

### log10 transformation for non-normal data
BD$lgage<-log10(BD$Age) # ! check for normality again after transformation before proceeding !
# calculate uPI.lgage and lPI.lgage in d same way as shown above
cbind(10^(lPI.lgage),10^(uPI.lgage)) # reverses the transformation

### transform Tukey (alternative to log10)
BD$Age.t = transformTukey(BD$Age, plotit=TRUE) # auto does normality tests,, v stonks; transformation of data done depends on lambda value:
## if (lambda >  0){TRANS = x ^ lambda} 
lPI.age2 <- lPI.aget^(1/lambda)
uPI.age2 <- uPI.aget^(1/lambda)
## if (lambda == 0){TRANS = log(x)} 
lPI.age2 <- exp(lPI.aget)
uPI.age2<- exp(uPI.aget)
## if (lambda < 0){TRANS = -1 * x ^ lambda}
lPI.age2 <- (-1/lPI.aget)^(1/absolute_value_of_lambda)
uPI.age2<- (-1/uPI.aget)^(1/absolute_value_of_lambda)
# perform above calcs again to get lPI.aget, uPI.aget then reverse transform as seen above

## hypothesis testing
### one sample t-test for comparing 2 means (population sd unknown)
# if data is not normally distributed, acknowledge it and say: However, given that the samples are large enough (n>30), using CLT, we can assume that the sampling distribution of means will be approximately normal. Hence we can use t.test to compare the means. 
t.test(BD$Age, alternative="two.sided", mu=35, conf.level = 0.95) # alternative can be "less" or "more"
# do not rej H0 conclusion: Based on the results (t=-1.12 & p-value>0.05), we do not have sufficient evidence to reject H0. Hence our sample data does not provide sufficient evidence to accept that population mean age is significantly different from 35 at the 5% level of significance.
# rej H0 conclusion: Based on the results (t=-10.46 & p-value<0.05), we have sufficient evidence to reject H0 and can accept that the population mean age is significantly less than 40 at the 5% level of significance.

### two sample t-test for comparing means (population sd unknown)
t.test(data$time~data$company, alternative="greater") # assuming sd are unequal; if sd assumed to be equal then add var.equal=T
t.test(data$before, data$after, paired=T) # test for diff in means of paired samples

### z-test for proportion
age50 <- BD %>% filter(Age>50)
p50 <- nrow(age50)/nrow(BD)
z <- (p50 - 0.18) / sqrt(0.18*(1-0.18)/nrow(BD)) # compute z-statistic for proportion
z
cv.age50 <- qnorm(0.05) # compute critical value
cv.age50
# do not rej H0 conclusion: From our results (z-statistic=-0.50 & z-critical=-1.64), the z-statistic is lying in the non-rejection region. Thus we have insufficient evidence to reject H0 and we conclude that (use H0).
# rej H0 conclusion: From our results (z-statistic=-3.98 & z-critical=-1.64), the z-statistic is lying in the lower critical region. Thus we have sufficient evidence to reject H0 and accept that proportion of Age is statistically less than 0.18 at the 5% level of significance.

### welch-anova test for comparing >= 2 means
# H0: Mean number of cigarettes smoked per day is same across people of the three age groups; 
# H1: At least one age group has a different mean number of cigarettes smoked per day from the other age groups.
wa.out1 <- SK %>% welch_anova_test(cigs ~ agegp)
gh.out1 <- games_howell_test(SK, cigs ~ agegp)
wa.out1
gh.out1




# LECT & TUT 6 : Linear Regression
### Do not perform any data transformation in the linear regression analysis, unless requested for FINALS

## cleaning d data (Not Tested for Finals in Linear Regression)
d2_new$logcases <- replace(d2_new$logcases, is.infinite(d2_new$logcases), NA) # replace inf values with NA
signif(value, 3) # round value to 3 sf
round(value, 2) # round value to 2 dp
pivot_longer(data) # convert wide to long form data
pivot_wider(data) # convert long to wide form data

## linear regression
fit1 = lm(Human.Capital.Index~loggdp, newgdp) # DV~IV; use '+' to insert multiple IVs
summary(fit1) # R-squared measures proportion of variance explained by model, varies between 0 and 1, larger value means better fit; p-value under coeff is for t-test with H0: coeff = 0
summary(fit1)$coeff 
fit1$coefficients[1] # to access each individual coefficient
# interpreting coeff for model with log(variables): Slope(b1) = 10.7. When log-Health expenditure of a country increases by 1 unit, we expect to see an average increase of Life Expectancy At Birth by 10.7. OR The exponential of the intercept, 0.0713, is the mean number of new cases of Covid-19 per million people assuming GDP per capita and population density are both zero. The exponential of the coefficient for d2_new$loggdp, 2.86, represents the expected change in number of new confirmed Covid-19 cases per million people given a unit increase in log10 of GDP per capita, with population density held constant. The exponential of the coefficient for d2_new$logpop, 1.03, represents the expected change in number of new confirmed Covid-19 cases per million people given a unit increase in log10 of population density, with GDP per capita held constant.

## logistic regression (categorical DV)
fit_log1 <- glm(Survived ~ Sex + Age, family="binomial", titanic) # variables can be factor or numeric
# Our logistic regression model is: logit(p) = log (p/(1-p)) = beta0 + beta1 rules + beta2 age + beta3 tserved + beta4 married + beta5 priors + beta6 black
# interpretation of coeff: b0 predicts log-odds of Survived=1 when all variables (Sex and Age) = 0; bi predicts expected change in log-odds of Survived=1 when [categorical Xi (Sex) becomes 1] OR [per unit increase in continuous Xi (Age)], holding the other variables constant 

## predicting using lm
predict(fit_log1, newdata = data.frame(Sex="male", Age=35))
## predicting using glm
newClient = data.frame(rules = 0, age = 55*12, tserved = 71, married = 1, priors = 0, black = 1)
predict(fit_supervisedRelease, newdata = newClient, type = 'response') # CHECK THE type = response part !!!!!

## Checking Assumptions 
# For residual plot after regression fitting, we usually plot "residuals against X" or "residuals against Fitted values y_hat".
# Things you should check here:
# 1. Mean-zero error. If residual plot scatters around y = 0 horizontal line?
# 2. Homoscedasticity. If any fan-shape residual plot?
# 3. Independently distributed error. If random-looking residual pattern?
# 4. Normally distributed error. ---> Q-Q Plot , Any deviation from 45 degree line?
# 5. Linearity assumption between y and x; X-Y scatterplot shows linear relation?

# Use Q-Q plot to test normally distributed error assumption. A normally looking sample data should scatter along the 45 degree line.
plot(lm_traffic, which = c(1,2), caption = list("Resid vs. Fitted", "Normal Q-Q")) # lm_traffic is the linear regression model
pairs.panels(iris) # check for linearlity, possible collinearlity



# LECT & TUT 7 : Log Regression & Time Series

## smoothing using simple moving average window
SMA(df$y, n=3) # where n refers to size of window 

## simple k-period moving average model 
df$sma2 = SMA(df$y, n=2) # taking window size of 2
df$sma_predict = dplyr::lag(df$sma2,1) # lagging by 1; assumes auto-correlation without trends, cycles or seasons; used for short range forecasting assuming things will not change

## exponential smoothing models
### preparing d data
uscases = usdata %>% select(new_cases_per_million)
n = nrow(uscases)-31
us_train = uscases[1:n,1] # taking out the training data set
n1 = n+1
us_test = uscases[n1:nrow(uscases),1] # preparing testing data set
us_train.ts = ts(us_train, start=1) # ts function from TTR package converts df to ts object bc HoltWinters function used below requires ts object

### building d model
hw = HoltWinters(us_train.ts, beta=F, gamma=F) # single exponential smoothing, no trend and seasonality
hw = HoltWinters(us_train.ts, gamma=F) # double exponential smoothing, have trend no seasonality           # Usually this is asked
hw = HoltWinters(us_train.ts) # holt-winters (triple exponential smoothing), have trend and seasonality

# •	Trends refer to a gradual upwards/downwards movement of a time series over time 
# •	Seasonal effects refer to effects that occurs/repeats at a fixed interval 
# •	Cyclical effects refer to longer-term effects that don’t have a fixed interval/length


### prediction 
hw_pred <- predict(hw, n.ahead = 4) # n.ahead refers to number of period to predict for
plot(hw, hw_pred) # plots line graph for original data and smoothed+predicted data on same axes
rmse = sqrt(mean((us_test-t(hw_pred[1:31]))^2)) # evaluating d prediction using test data set # root mean square error

## regression-based time series models
lm(subsetd$new_cases_per_million ~ subsetd$new_cases_per_million_lagged * subsetd$continent) # predicting new_cases_per_million based on previous new_cases_per_million (new_cases_per_million_lagged), continent (Asia (default), Europe (variable1), North America (variable2)) and their interaction
# interpretation of coeff: The intercept, 5.75 (2dp), is the mean number of new cases of Covid-19 per million people assuming continent is Asia, and new_cases_per_million_lagged is zero. The coefficient for new_cases_per_million_lagged, 0.522 (3sf), is the expected average change in the number of new Covid-19 cases per million people for every unit increase in new_cases_per_million_lagged, assuming continent is Asia. The coefficients of continentEurope, -0.703 (3sf) and continentNorth America, -12.98 (2dp) are the average difference in number of new cases of Covid-19 per million people in Europe and North America respectively, compared to Asia, when new_cases_per_million_lagged is zero. The coefficient of new_cases_per_million_lagged * continentEurope, -0.0957 (3sf), is the average difference in change in new_cases_per_million between Europe and Asia for a unit increase in new_cases_per_million_lagged. The coefficient of new_cases_per_million_lagged * continentNorth America, 0.330 (3sf), is the average difference in change in new_cases_per_million between North America and Asia for a unit increase in new_cases_per_million_lagged.

## model selection
### using anova to compare one subset of a model to the model
m_full <- lm(y~x1 + x2, df1)
m_restricted <- lm(y~x1, df1)
anova(m_restricted, m_full) # H0: coeff of x2 = 0, H1: coeff of x2 != 0
# conclusion: The p-value is very large and > 0.05. We cannot reject H0. The ANOVA suggests that the full model is not significantly better than the restricted model. Thus, the simpler model (restricted model) should be used.

### using stepwise regression
step(m_full, direction = "backward") # start with full model and eliminate predictors one by one; can set direction to "forward" or "both" also

## detecting multicollinearity 
car::vif(lm(y~x4 + x5 + x6, dfMC)) # if values for any variables > 5 => they are highly correlated



# LECT & TUT 8

## partitioning data set (to prevent overfitting of model)
set.seed(1) # for reproducibility
df$partitionNum <- sample(1:3, size=nrow(df), prob=c(0.6,0.2,0.2), replace=T) # split data randomly into 3 (60:20:20)
df$partition <- factor(df$partitionNum, levels=c(1,2,3), labels=c("Train", "Valid", "Test"))
df_train <- df %>% filter(partition=="Train")
df_valid <- df %>% filter(partition=="Valid")
df_test <- df %>% filter(partition=="Test")

## pca
pca1 <- prcomp(mtcars1, center=T, scale=T)
summary(pca1) # gives sd, proportion of variance and cumulative proportion for each PC
pca1$rotation[,1:2] # examining loadings on first 2 PCs
mtcars$pc1 <- pca1$x[,"PC1"] # extracting the first PC

d2 <- d2 %>% mutate(pc1 = pca$x[,"PC1"],pc2 = pca$x[,"PC2"], pc3 = pca$x[,"PC3"], pc4 = pca$x[,"PC4"], pc5 = pca$x[,"PC5"])
trainset = d2 %>% filter(partition=="Train")
testset = d2 %>% filter(partition=="Test")
logreg = glm(class ~ pc1+pc2+pc3+pc4+pc5, family="binomial", trainset)
summary(logreg) # tells us which PCs are significant predictors
testset$pred = predict(logreg, testset, type="response") # type="response" ensures that probabilities are returned instead of logits

## k-means clustering 
### finding optimal number of clusters to use
set.seed(1)
wss <- rep(NA,20)
for(k in c(2:20)) {
  wss[k] = kmeans(whX, k, nstart = 10)$tot.withinss
}
plot(wss, type ="b", xlab = "Number of clusters", ylab = "Total within-cluster sum of squares") # choose the number of clusters that corresponds to the elbow in the graph

### visualizing the cluster
set.seed(1) # fix random number generator bc k-means depends on random initialisation => ensures that everyone can get same results from the kmeans below
km_obj <- kmeans(whX,3)
fviz_cluster(km_obj, whX)
km$center # gives the centers of each cluster
km$cluster # gives which cluster each data point belongs to


### Accuracy, Sensitivity, Precision & Specificity
## TP = True Positive, TN = True Negative(Predicted No, Actual Yes), FP = False Positive(Predicted Yes, Actual No), FN = False negative
# Calculation for accuracy
(TP+TN)/(TP+FN+FP+TN)
# Calculation for sensitivity
TP/(TP+FN)
# Calculation for precision
TP/(TP+FP)
# Calculation for specificity
TN/(FP+TN)


# LECT & TUT 10 & 11 (skip 9)

## HTML table format 
Maximize total reach using decision variables $X_1$, $X_2$, $X_3$ | Reach = 500 $X_1$ + 2000 $X_2$ + 300 $X_3$
--- | --- 
Subject to |  
Budget Constraint | 100$X_1$ + 250$X_2$ + 50$X_3$ $\leq$ 5000
Radio Ad Limit | $X_1$ + $\quad$ + $\quad$ $\leq$ 40
Newspaper Ad Limit | $\quad$ + $X_2$ + $\quad$ $\leq$ 10
Social Media Ad Limit | $\quad$ + $\quad$ + $X_3$ $\leq$ 80
Non-Negativity Constraint 1 | $X_1$ + $\quad$ + $\quad$ $\geq$ 0
Non-Negativity Constraint 2 | $\quad$ + $X_2$ + $\quad$ $\geq$ 0
Non-Negativity Constraint 3 | $\quad$ + $\quad$ +$X_3$ $\geq$ 0
# Binary, Integer, Non-negativity Constraints | $X_1$ to $X_3$ all binary, integers and $\geq$ 0

## linear optimisation
objective.fn = c(1000, 1200, 1500)
const.mat = matrix(c(30, 40, 50,
                     20, 50, 40,
                     30, 10, 15), ncol=3, byrow=T) # ncol is number of decision variables 
const.dir = c(rep(">=", 3))
const.rhs = c(5000, 3000, 2500)
lp.solution = lp("min", objective.fn, const.mat, const.dir, const.rhs, compute.sens=T)
lp.solution # gives optimal value of objective function
lp.solution$solution # gives values of variables needed to achieve d above value
lp.solution$duals # gives shadow price, ie change in obj fn value per unit-increase in value of constraint, holding all else constant

### sensitivity analysis (made possible by compute.sense=T in lp())
range.objcoef = cbind(lp.solution$sens.coef.from, lp.solution$sens.coef.to)
rownames(range.objcoef) = c('coef x1', 'coef x2')
colnames(range.objcoef) = c('from', 'to')
print(round(range.objcoef, 1))


## integer optimisation (Binary)
objective.fn <- c(6, 7, 9, 8, 10, 20)
const.mat <- matrix(c(2, 4, 3, 3, 4, 5, # 12-credit
                      1,-1, 0, 0, 1, 0, # Pre-Requisite for BT2
                      0, 0, 1,-1, 0, 0, # Pre-Requisite for CS2
                      1, 0, 1, 0, 0,-2), # Pre-Requisite for DT
                      ncol=6 , byrow=TRUE) 
const.dir <- c(rep(">=", 4))
const.rhs <- c(12, 0, 0, 0)
#solving model
lp.solution <- lp("min", objective.fn, const.mat, const.dir, const.rhs, binary.vec = c(1:6))
lp.solution$solution #decision variables values  # use binary.vec to specify binary variables; int.vec used to specify integer variables 

## integer optimisation
objective.fn <- c(12.4, 13.75, 12.3, 12.15, 18.35,
                  8.5, 18.55, 8.80, 9.85, 16.45,
                  9.3, 14.25, 6.45, 11.15, 14.95)
const.mat = matrix(c(# Constraint 1: Capacity constraint
                     rep(1,5),rep(0,10),
                     rep(0,5),rep(1,5),rep(0,5),
                     rep(0,10),rep(1,5),
                     # Constraint 2: Demand constraint
                     rep(c(1,0,0,0,0) , 3),
                     rep(c(0,1,0,0,0) , 3),
                     rep(c(0,0,1,0,0) , 3),
                     rep(c(0,0,0,1,0) , 3),
                     rep(c(0,0,0,0,1) , 3)
                     ), ncol = 15, byrow = TRUE)
const.dir = c(rep('<=', 3), rep('>=', 5))
const.rhs = c(1300,900,400,160,330,600,590,880)
# solving the linear problem
lp.solution = lp(direction = 'min', objective.fn, 
              const.mat, const.dir, const.rhs,all.int = T, # all.int = T means all are integers 
              compute.sens = TRUE)
matrix(c(lp.solution$solution), ncol = 5, byrow = T)
print(lp.solution)



## Rmd code block: change the names!
```{r q1a, echo=TRUE}

```