function [K]= covGP_squaredEx(x,x2, sigma, lengp)
    d= pdist2(x, x2,'euclidean');
    [K]= squaredExp(d,sigma,lengp);

    
function [k]= squaredExp(d,sigma,lengp)
% Squared Exponential covarience function
k=sigma*exp(-d.^2/(2*lengp^2));