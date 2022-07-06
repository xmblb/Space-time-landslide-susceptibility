##set the working directory where data files are located and where output files will be saved:
# WORK=getwd() #define your own working directory (containing data files), such as WORK="~/my/working/directory/"
WORK = getwd()
setwd(WORK)
library(INLA)
## load data from file: 
data.new = read.table("MatrixTaiwan.txt", sep="\t", header=TRUE)


#define penalized complexity prior specifications that we will use in the model formulas: 
hyper.iid = list(theta1 = list(prior="pc.prec", param=c(0.1, 0.5)))
hyper.rw = list(theta1 = list(prior="pc.prec", param=c(0.1, 0.5)))

##################################
###create the formula for model###
##################################
formula1 = Label ~ -1 + intercept + SlopeStd  + ProfileStd + NorthStd + EastStd +
  f(data.new$Lithology, model="iid", hyper=hyper.iid, constr=T)+ 
  f(data.new$SlopeM, model="rw1", hyper=hyper.rw, constr=T) + 
  f(data.new$PlanM, model="rw1", hyper=hyper.rw, constr=T) + 
  f(data.new$ProfileM, model="rw1", hyper=hyper.rw, constr=T)+ 
  f(data.new$NorthM, model="rw1", hyper=hyper.rw, constr=T)+ 
  f(data.new$EastM, model="rw1", hyper=hyper.rw, constr=T)+ 
  f(data.new$RainMean, model="rw1", hyper=hyper.rw, constr=T)+ 
  f(data.new$NDVI3, model="rw1", hyper=hyper.rw, constr=T)

model=inla(formula1,
           family="binomial",
           data=data.new, 
           control.fixed=list(prec=.1),
           num.threads=2,
           control.inla = list(int.strategy = "eb"),
           control.compute = list(cpo=TRUE, dic=TRUE, waic=TRUE, config=TRUE),
           control.predictor=list(compute=TRUE, link=1),
           inla.mode = "experimental",
           verbose = TRUE
)

## goodness of fit
library(pROC)
dataLabel = data.new$Label
predictedSus = model$summary.fitted.values$mean
ROC.fit = roc(dataLabel~predictedSus)
print(ROC.fit$auc)








