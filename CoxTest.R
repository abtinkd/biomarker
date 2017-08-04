library(survival)
library(entropy)
options(digits = 4)

raw.dat <- read.csv("Biomarker_Data_Fern_8_29_16.csv")
raw.dat$sex <- as.factor(raw.dat$sex)
raw.dat$raceb <- as.factor(raw.dat$raceb)
raw.dat$smoker <- as.factor(raw.dat$smoker)
raw.dat$htn_med <- as.factor(raw.dat$htn_med)
raw.dat$chol_med <- as.factor(raw.dat$chol_med)
raw.dat$dm_med <- as.factor(raw.dat$dm_med)
raw.dat$dm_hx <- as.factor(raw.dat$dm_hx)
raw.dat$chf_hx <- as.factor(raw.dat$chf_hx)
raw.dat$chd_hx <- as.factor(raw.dat$chd)
raw.dat$stroke_hx <- as.factor(raw.dat$stroke_hx)


ttodeath <- dat$ttodeath
death <- dat$death
sbp <- dat["sbp"]
surv <- Surv(ttodeath,death)
mod <- coxph("surv~sbp", ties="breslow")

# res2.train <- read.csv("train_set2.csv")
# res2.test <- read.csv("test_set2.csv")
dat <- raw.dat[complete.cases(raw.dat[c("sbp","death","ttodeath")]),]

spec = c(train = .6, test = .2, validate = .2)
g = sample(cut(
  seq(nrow(dat)),
  nrow(dat)*cumsum(c(0,spec)),
  labels = names(spec)
))
res = split(dat, g)
cn <- colnames(dat)

fields <- c("sbp")
run.it(cn[5:length(cn)-2], 1.5)
run.it(c("sex", "age", "sbp", "bmi", "crp", "trigly", "creat", "chf_hx", "wlk_spd"), 2.5)
run.it(c("sbp"), 1)

run.it <- function(fields, scl) {
  f.vars <- paste(fields, collapse="+")
  f.fmla <- as.formula(paste("Surv(peryr_exm, mortstat)~",f.vars, sep=""))
  cx <- coxph(f.fmla, data=res$train, model=TRUE)
  print(summary(cx))
  #par(mfrow=c(2,2))
  print("Training")
  plot.cmp("Training", cx, res$train, 5, fields, scl)
  print("Testing")
  plot.cmp("Testing", cx, res$test, 5, fields, scl)
  print("Validation")
  plot.cmp("Validation", cx, res$validate, 5, fields, scl)
}

plot.cmp <- function(ttle, cx, dat, tst.time, fields, scl) {
  base <- basehaz(cx)
  tfit <- survfit(cx, newdata = dat[c(fields, c("mortstat", "peryr_exm"))])
  tst.val <- base$hazard[base$time==tst.time]*scl
  tme.mask <- tfit$time==tst.time

  boxplot(abs((tfit$cumhaz)[tme.mask,][dat["mortstat"]==1]),
          abs((tfit$cumhaz)[tme.mask,][dat["mortstat"]==0]),
          names=c("Died", "Survived"),
          ylim=c(0,min(2, max(abs((tfit$cumhaz)[tme.mask,][dat["mortstat"]==1])))),
          main=paste(ttle, paste(fields, collapse=" "), sep="\n"), xlab="Index", ylab="Diff")
  abline(tst.val, 0, lty=2)

  tmp <- c(sum(abs((tfit$cumhaz)[tme.mask,][dat["mortstat"]==1])>=tst.val),
           sum(abs((tfit$cumhaz)[tme.mask,][dat["mortstat"]==1])<tst.val))
  result <- matrix(c(tmp, tmp[1]/sum(tmp)), ncol=3, byrow = TRUE)
  colnames(result) <- c("Correct", "Incorrect", "Pct")
  tmp <- c(sum(abs((tfit$cumhaz)[tme.mask,][dat["mortstat"]==0])<tst.val),
           sum(abs((tfit$cumhaz)[tme.mask,][dat["mortstat"]==0])>=tst.val))
  result <- rbind(result, c(tmp, tmp[1]/sum(tmp)))
  result <- rbind(result, c(colSums(result[,1:2])/sum(result[,1:2]),
                            sum(result[2,1:2])/sum(result[,1:2])))
  rownames(result) <- c("Died", "Survived", "Pct")
  print(result)
}


confMat(cx, predHaz(cx, res$train))
confMat(cx, predHaz(cx, res$test))
confMat(cx, predHaz(cx, res$validate))

pred.train <- cbind(predict(cx, newdata = res$train, type="expected"), res$train[c("mortstat","peryr_exm", "age", "sbp")])
colnames(pred.train)[1]<-"prediction"
confMat(cx, pred.train)

pred <- cbind(predict(cx,newdata=res$test, type="expected"), res$test[c("mortstat","peryr_exm", "age", "sbp")])
colnames(pred)[1]<-"prediction"
head(pred[pred[,2]==1,][order(pred[pred[,2]==1,][,1], decreasing=FALSE),])
head(pred[order(pred[,1], decreasing=TRUE),], n=25)

pred.fit <- predict(cx, newdata=res$test, type="expected",se.fit=TRUE)
sum(pred.fit$se.fit, na.rm = TRUE)
pred.fit <- cbind(pred.fit$fit, pred.fit$se.fit, res$test)
colnames(pred.fit)[1] <- "predict"
colnames(pred.fit)[2] <- "SE"
pred.fit[pred.fit$mortstat==1,][order(pred.fit[pred.fit$mortstat==1,]$predict),]


# Use predict() to evaluate risk
# This requires "mortstat" and "peryr_exm" values
predHaz <- function(mod, df) {
  pred <- predict(cx, newdata=df, type="expected", se.fit = TRUE)
  pred <- cbind(pred, df[c("mortstat","peryr_exm", "age", "sbp")])
  colnames(pred)[1]<-"prediction"
  colnames(pred)[2]<-"stdErr"
  return(pred[complete.cases(pred),])
}

confMat <- function(mod, df) {
  base <- basehaz(mod)
  mask <- df$prediction < base[base$time==4,]$hazard
  print( matrix( c(
    sum(1-df[!mask,]$mortstat, na.rm = TRUE), sum(1-df[mask,]$mortstat, na.rm = TRUE),
    sum(df[!mask,]$mortstat, na.rm = TRUE), sum(df[mask,]$mortstat, na.rm = TRUE)
  ),
  nrow = 2, ncol = 2, byrow = TRUE))
}

entropy(table(res$train$mortstat), method="ML")

ent.dat <- table(res2.train$mortstat)/length(res2.train$mortstat)
entropy.empirical(ent.dat, unit="log2")

table(res$train$mortstat)

head(survfit(cx, newdata = res$train[c("sbp")]))

all.fit <- survfit(cx, newdata = res$train[c("sbp")])
all.haz <- cbind.data.frame(all.fit$time, all.fit$cumhaz)
haz.cmp <- merge(all.haz, base, by.x="all.fit$time", by.y="time")
mean(haz.cmp$`all.fit$cumhaz`-haz.cmp$hazard)

dim(all.fit$cumhaz - base$hazard)
dim(colMeans(all.fit$cumhaz - base$hazard))
haz.cmp <- cbind.data.frame(res$train$mortstat, all.fit$cumhaz[all.fit$time==4] - base$hazard[base$time==4])
colnames(haz.cmp) <- c("mortstat", "hazDif")
head(haz.cmp[order(haz.cmp$hazDif),], n=25)
head(haz.cmp[haz.cmp$mortstat==1,][order(haz.cmp[haz.cmp$mortstat==1,]$hazDif),], n=25)
head(haz.cmp[haz.cmp$mortstat==0,][order(haz.cmp[haz.cmp$mortstat==0,]$hazDif),], n=25)
all.fit$n.censor

base <- basehaz(cx)
base.fit <- survfit(cx)
head(base)

all.test <- as.data.frame(cbind(all.fit$cumhaz[all.fit$time==5], res$train$mortstat, res$train$peryr_exm))
colnames(all.test) <- c("cumhaz", "mortstat", "time")
head(cbind(all.test, pred.train$prediction))

predict(cx, newdata = res$train, type="expected")

dim(all.test)
head(all.test[order(all.test$cumhaz),], n=25)
mean(all.haz[all.fit$time == 4,])
base[base$time==4,]

dim(all.haz)
dim(res$train)

base <- basehaz(cx)
tpred <- predict(cx, newdata = res$train[c("sbp","age","peryr_exm", "mortstat")], type="expected") # 0.358758 0.043292
tfit <- survfit(cx, newdata = res$train[c("sbp","age","peryr_exm", "mortstat")])


head(cbind.data.frame(tfit$time, tfit$cumhaz[,1], tfit$cumhaz[,2], tfit$cumhaz[,3], tfit$cumhaz[,4], base$hazard))
plot(abs((base$hazard-tfit$cumhaz)[tfit$time==4,][res$train["mortstat"]==1]),
     main="Died", xlab="Index", ylab="Diff")
plot(abs((base$hazard-tfit$cumhaz)[tfit$time==4,][res$train["mortstat"]==0]),
     main="Survived", xlab="Index", ylab="Diff")
boxplot(abs((tfit$cumhaz)[tfit$time==7,][res$train["mortstat"]==0]),
     main="Survived", xlab="Index", ylab="Diff")


merge(cbind.data.frame(tpred, res$train[c(1,2,3,5),c("sbp","peryr_exm", "mortstat")]),
      base, by.x = "peryr_exm", by.y = "time", all.x = TRUE)
head(cbind.data.frame(tfit$time, tfit$cumhaz[,1], tfit$cumhaz[,2], tfit$cumhaz[,3], tfit$cumhaz[,4], base$hazard))


