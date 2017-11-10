
reset;
option solver "ampl/loqo";


# loading iris data (1st class only)
model iris_data_2.mod;



# variables
var r >= 0;
var C {1..dim};

minimize ball: r;
subject to c1 {i in 1..n_feat}: (sum{d in 1..dim} ((X[i,d] - C[d]) ^ 2)) <= (r * r);


solve;
display C;
display r;
