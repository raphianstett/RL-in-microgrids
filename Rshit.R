setwd("~/Studium/Bachelorarbeit/Coding")
dat <- read.csv("household_with_pv.csv", sep = ";")
head(dat)

dat[is.na(dat)]<- 0


# plot mean consumption over one day
consumption <- dat$Consumption
consumption[is.na(consumption)] <- 0

cons <- rep(0, each = 24)

for(i in 1:364){
  for(j in 1:24){
    print(consumption[i*24 + j])
    
    cons[j] <- cons[j] + consumption[i * 24 + j]
    
  }
}

con_mean <- cons / 365

plot(con_mean)

# plot mean consumption over the year

cons2 <- rep(0, each = 365)

for(i in 1:364){
  for(j in 1:24){
    
    cons2[i] <- cons2[i] + consumption[i+j]
  }
}

plot(cons2, type = "l")


# plot mean production over year

production <- dat$Production

production[is.na(production)] <- 0

prod <- rep(0, each = 365)

for(i in 1:364){
  for(j in 1:24){
    
    prod[i] <- prod[i] + production[i+j]
  }
}

plot(prod, type = "l")

# plot mean production of one day

prod2 <- rep(0, each = 24)

for(i in 1:364){
  for(j in 1:24){
    
    prod2[j] <- prod2[j] + production[i * 24 + j]
    
  }
}

prod_mean <- prod2 / 365

plot(prod_mean, type = "l")
lines(con_mean)
plot(con_mean)

