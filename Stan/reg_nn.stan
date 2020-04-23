data {
  int<lower=0> n;  // Number of observations
  int<lower=0> d;  // Dimension of the matrix
  int<lower=0,upper=d> h; // Number of neurons
  matrix[n,d]  X;  // Intput matrix
  vector[n] y;     // Output observations
}                  
parameters {       
  real  b;           //mean constant value
  matrix[d,h] u;       // regresion coefficients
  vector[h] a;       //Independent variables input
  vector[h] v;       //constant  layers 
  real<lower=0>sigma;//variance value
}
transformed parameters{
  vector[n] mu;
  for (i in 1:n){
    mu[i] = b;
    for(j in 1:h) mu[i] += v[j]*tanh(a[j] + X[i,]*u[,j]);
  } 
}
model {
  // priors
  target += normal_lpdf(b|0,1);
  target += normal_lpdf(to_vector(u)|0,1);
  target += normal_lpdf(a|0,1);
  target += student_t_lpdf(sigma|4,0,1);
  // likelihood
  target += normal_lpdf(y|mu,sigma);
}
generated quantities{
  vector[n] fit;
  vector[n] log_lik;
  vector[n] residuals;
  for (i in 1:n){
    fit[i] = normal_rng(mu[i],sigma);
    log_lik[i] = normal_lpdf(y[i]|mu[i],sigma);
  }
  residuals = y - fit;
}
