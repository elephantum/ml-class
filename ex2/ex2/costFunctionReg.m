function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_x = sigmoid(X * theta);

sum_y1 = y' * log(h_x);
sum_y0 = (1-y') * log(1 - h_x);

theta_all_but_first = [0; ones(size(theta,1) - 1, 1)] .* theta;

J = -1 / m * (sum_y1 + sum_y0) + lambda / 2 / m * theta_all_but_first'*theta_all_but_first;

grad = 1/m * X' * (h_x - y) + lambda / m .* theta_all_but_first;

% =============================================================

end
