%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X]

%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.1;
num_iters = 500;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J1] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Try different values of learning rate
%alpha = 0.3;
%theta2 = zeros(3, 1);
%[theta2, J2] = gradientDescentMulti(X, y, theta2, alpha, num_iters);

%alpha = 1;
%theta3 = zeros(3, 1);
%[theta3, J3] = gradientDescentMulti(X, y, theta3, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J1), J1, 'b');
xlabel('Number of iterations');
ylabel('Cost J');
% Plot the convergence graph for each value of learning rate
%hold on;
%plot(1:numel(J2), J2, 'r');
%plot(1:numel(J3), J3, 'k');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
price = [1 , (1650-mu(1))/sigma(1) , (3-mu(2))/sigma(2)]*theta; 

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================
%% Analytical solution through ordinary least squares

fprintf('Solving with normal equations...\n');

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
price = [1 , 1650 , 3]*theta; 

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

