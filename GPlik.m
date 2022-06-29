 function  lik=GPlik(theta,x,y)
%% returns marginal likelihood  
 
[K]= covFsqE(x,x, theta,0);
[lik]= likGP(K,y');
% For tghe maximum likelihood , invert the sign.
lik=-lik;

%%
function [lik]= likGP(K,y)

% Calculates log-marginal likelihood using
% gaussian process
%
% K= covariance matrix: k(x,x)
% K2= k(x, xtest)
% K22= k(xtest, xtest)
% 
% ref: 
%    M. Bishop,  Pattern recog. and machine learn.
%    Rusmussena and Williams,  Gaussian process for machine learning

% Choleski decomposition with "lower" option
L= chol(K,'lower');    % size =  [observation x observation]
alpha= L'\(L\y);        % size = [observation x dimension]

% Log marginal likelihood
lik=-0.5*y'*alpha-sum(log(diag(L)))-log(2*pi)*size(K,1)/2;
