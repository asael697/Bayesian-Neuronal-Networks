data {
  int<lower=0> n; //Numero de observaciones
  int<lower=0> d; //dimension de la matriz 
  matrix[n,d] X;  //matriz con las variables explicativas
  vector[n] y;    //ariable de respuesta
}
parameters {
  real beta0;
  vector[d] beta;               // vector de coeficientes
  real<lower=0> sigma;          // varianza de y 
  vector<lower=0>[d] lambda;    // matriz diagonal D
  cholesky_factor_corr[d] Omega;// matriz de correlaciones
}
transformed parameters{
  matrix[d,d] Sigma;          // Matriz de covarianza para beta
  vector[n] mu = beta0+ X*beta;
  // Transformacion de la matriz Sigma
  Sigma = diag_pre_multiply(lambda,Omega);
}
model {
  // Prioris
  target += cauchy_lpdf(sigma|0,5);
  target += student_t_lpdf(beta0|4,0,1);
  target += lkj_corr_cholesky_lpdf(Omega|7);
  target += student_t_lpdf(lambda|4,0,5);
  target += multi_normal_cholesky_lpdf(beta|rep_vector(0,d),Sigma);
  // loglikelihood
  target += normal_lpdf(y|mu,sigma);
}
generated quantities{
  matrix[d,d] Sigma_real = multiply_lower_tri_self_transpose(Sigma);
    vector[n] fit;
  vector[n] log_lik;
  vector[n] residuals;
  for (i in 1:n){
    fit[i] = normal_rng(mu[i],sigma);
    log_lik[i] = normal_lpdf(y[i]|mu[i],sigma);
  }
  residuals = y - fit;
}
