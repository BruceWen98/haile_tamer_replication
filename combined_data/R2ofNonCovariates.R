library(dplyr)
library(gt)
library(stargazer)

data <- read.csv(("/Users/brucewen/Desktop/honors_thesis/estimation/combined_data/out_dicts_ALL.csv"), header=TRUE)

## Regressions
reg1 <- lm(n ~ high_estimate + cat0 + cat1 + loc + final_price, data=data)

stargazer(reg1,
    column.labels=c('reg1'),
    type="text", title="OLS Regressions", out="R2ofN_onCovariates.txt")