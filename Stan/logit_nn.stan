functions {
  
  // Calculate values for layers
  matrix calc_u(matrix X, matrix weights, vector bias) {
    vector[rows(X)] bias_unit;
    matrix[rows(X), cols(weights)] res;
    
    for(i in 1:rows(X)) bias_unit[i] = 1;
    res = append_col(X, bias_unit) * append_row(weights, bias');
    return res;
  }
  // Calculate values for layers (alternative)
  matrix calc_u_alt(matrix X, matrix weights, vector bias) {
    matrix[rows(X), cols(weights)] res;
    
    for(j in 1:cols(weights))res[,j] = X*weights[,j] + bias[j];
    return res;
  }
}
data {
  int<lower=1> N;
  int<lower=1> inputs;
  int<lower=1> outputs;
  matrix[N, inputs] X;
  int y[N];
  int<lower=0> nodes_h1;
  int<lower=0> nodes_h2;
}
parameters{
  
  // Weights
  matrix[inputs, nodes_h1] w_in1;
  matrix[nodes_h1, nodes_h2] w_12;
  matrix[nodes_h2, outputs] w_2out;
  
  // Biases
  vector[nodes_h1] b_1_raw;
  vector[nodes_h2] b_2_raw;
  vector[outputs] b_out_raw;
  
  real<lower=0> sigma_b_1;
  real<lower=0> sigma_b_2;
  real<lower=0> sigma_b_out;
  
}
transformed parameters{
  
  matrix[N, nodes_h1] v_1;
  matrix[N, nodes_h2] v_2;
  matrix[N, outputs] v_out;
  
  vector[nodes_h1] b_1;
  vector[nodes_h2] b_2;
  vector[outputs] b_out;

  b_1 = 0.0 + sigma_b_1*b_1_raw;
  b_2 = 0.0 + sigma_b_2*b_2_raw;
  b_out = 0.0 + sigma_b_out*b_out_raw;
    
  v_1 = tanh(calc_u_alt(X, w_in1, b_1));
  v_2 = tanh(calc_u_alt(v_1, w_12, b_2));
  v_out = calc_u_alt(v_2, w_2out, b_out);
  
}
model{
  
  // Output
  for(n in 1:N)
    y[n] ~ categorical_logit(to_vector(v_out[n,]));
  
  // Priors
  to_vector(w_in1) ~ normal(0, 1);
  to_vector(w_12) ~ normal(0, 1);
  to_vector(w_2out) ~ normal(0, 1);
  
  // Hyperpriors
  sigma_b_1 ~ double_exponential(0, 1);
  sigma_b_2 ~ double_exponential(0, 1);
  sigma_b_out ~ double_exponential(0, 1);

  // Aux
  b_1_raw ~ normal(0, 1);
  b_2_raw ~ normal(0, 1);
  b_out_raw ~ normal(0, 1);

}
