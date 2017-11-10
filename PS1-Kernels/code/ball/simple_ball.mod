
# ex4.mod
# Josselin MARNAT & Valentin Benozillo

reset;
option solver loqo;


# data 
param n_feat;
param dim;
param X {1..n_feat, 1..dim};

# variables
var r >= 0;
var C {1..dim};

minimize ball: r;
subject to c1 {i in 1..n_feat}: (sum{d in 1..dim} ((X[i,d] - C[d]) ^ 2)) <= (r * r);


data;
param n_feat = 4;
param dim = 2;
param X: 1 2 :=
1 0 0
2 2 0
3 0 2
4 1 1 ;

solve;
display C;
display r;
