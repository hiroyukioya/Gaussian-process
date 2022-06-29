function [K]= covFsqE(x,x2, param,mode)
% Constructs squared exponential covariance matrix
% NOTE:
%   x:   training data:   [feature dimension x training data observation] 
%   x2:  test data:   [feature dimenstion x test data observation]
% ///***************************************///
% param(1) = signal variance
% param(2) = length-scale parameter
% param(3) = noise variance
%            -> sigma term in [K+sigma*eye] in prediction equation

[a,b]=size(x);
[c,di]=size(x2);

d= pdist2(x', x2','euclidean');

if a~=c
    error ('dimension of x and x2 is different...');
end

if mode==0
    [K]= squaredExp(d,param);
elseif mode==1
    [K]= squaredExpTest(d,param);
end
    
    
function [k]= squaredExp(d,param)
    % Squared Exponential covarience function
    k = param(1)*exp(-d.^2/(2*param(2)^2));
    k2 = diag(diag(ones(size(k)).*param(3)));
    k=k+k2;

function [k]= squaredExpTest(d,param)
% Squared Exponential covarience function
k = param(1)*exp(-d.^2/(2*param(2)^2));
k=k;
    