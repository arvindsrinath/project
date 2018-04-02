clear ; close all; clc
input_layer_size  = 5;  
hidden_layer_size = 15; 
num_labels = 5;             
ite = 5;                         
X=csvread('sample.csv');
m = size(X, 1);
y = csvread('output.csv');
Z  = log10(X(:,2));
X(:,2) = Z;

new_size = m * random(ite);
new_size = ceil(new_size);
options = optimset('MaxIter', ite);

C = X(1:new_size,:);
vv = y(1:new_size,:);

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
lambda = 10;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, C, vv, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
K = X(new_size:m,:);
pred = predict(Theta1, Theta2,K );
pred1 = y(new_size:m,:);
vx=pred1;
csvwrite('1.csv',C);
%csvwrite('1.csv',pred);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == vx)) * 100);