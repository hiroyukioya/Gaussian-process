 function [lik, yhat, var, f] =GPpredict(x,y,test_points,theta)
 
[K]= covFsqE(x,x, theta,0);
[K2]= covFsqE(x,test_points, theta,1);
[K22]= covFsqE(test_points,test_points, theta,0);
[lik, yhat, var]= predictGP(K, K2, K22, y);

plotRPwSE(yhat, sqrt(var)*1.96, 1, test_points);hold on;
plot(x,y,'b+','markersize',8);
aaa=sprintf('%s%3.4g','Log marginal likelihood = ',lik);
title(aaa);

f.lik=lik;
f.yhat=yhat;
f.var=var;