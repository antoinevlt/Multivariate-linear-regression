function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. 

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
    
mu=mean(X);
sigma=std(X);
X_norm=(X-mu)./sigma;

% ============================================================

end
