

setwd("~/UNICAN/Redes Neuronales/data")

library(rmatio)


dat = read.mat("housing.mat")
datHousing = data.frame(t(dat$p))
datHousing$X14 = dat$t
names(datHousing) = c("tasa.criminalidad","prop.suelo.resid","prop.suelo.ind",
                      "rio","oxd.nitrico","hab","casa.mayor.40",
                      "dist.zona.trabajo","autovias","tasa.imp","prop.alumn.prof",
                      "pob.raza.color","nivel.econ","precio"
                      )


dat = datHousing
dat$rio = NULL
dat1 = datHousing[,-c(2,3,7,10,12)]
save.image("datHousing.Rdata")
