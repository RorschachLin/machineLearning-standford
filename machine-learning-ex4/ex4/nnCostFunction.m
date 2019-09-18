function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
% 25*401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
% 10*26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you   check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


y_matrix = eye(num_labels)(y,:);  % 5000*10
h = zeros(num_labels,1);
J_mTerm = 0;
p = predict(Theta1, Theta2, X);
% disp(sum(sum((Theta1(:,2:end).^2)),2));
%for i = 1:m
%	a1 = [1 X(i,:)]; % a1: 1*401
%	z2 = Theta1 * transpose(a1);   % z2: 25*1
%	a2 = sigmoid(z2);
%	a2 = [1;a2];  % 26*1
%	z3 = Theta2 * a2;
%	a3 = sigmoid(z3); % 10*1
%	h = a3;
%	J_KTerm = sum(-y_matrix(:,i) .* log(a3) - (1-y_matrix(:,i)) .* log(1-a3));
%	J_mTerm = J_mTerm + J_KTerm;
	%disp(log(1-a3));
%end


% ----------vectorized forward start------------
a1 = [ones(size(X, 1), 1) X];   % 5000*401
z2 = a1 * transpose(Theta1);    %5000*25
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2];  %5000*26
z3 = a2 * transpose(Theta2);   %5000*10
a3 = sigmoid(z3);
J_Term = -y_matrix .* log(a3) - (1-y_matrix) .* log(1-a3);
J_value = sum(sum(J_Term, 1), 2);
%disp(size(y_matrix));

% ----------vectorized forward end------------

% ----------regularized start------------
regTermL1 = sum(sum((Theta1(:,2:end).^2), 1),2);
regTermL2 = sum(sum((Theta2(:,2:end).^2), 1),2);
regTAll = lambda/(2*m) * (regTermL1 + regTermL2);
% ----------regularized end------------

J = (1/m) * J_value + regTAll;

% ----------------- bckprop start------------------
% disp(a1);
a1_vec_b = zeros(m, input_layer_size);
a2_vec_b = zeros(m, hidden_layer_size);
a3_vec_b = zeros(m, num_labels);
delta3 = zeros(num_labels, m);
delta2 = zeros(hidden_layer_size, m);
%for t = 1:m
%	a1_b = [1 X(t,:)]; % a1: 1*401
%	z2_b = Theta1 * transpose(a1_b);
%	a2_b = sigmoid(z2_b);
%	a2_b = [1;a2_b];  % 26*1
%	z3_b = Theta2 * a2_b;
%	a3_b = sigmoid(z3_b); % 10*1
%	%disp(size(a3_b));
%	a1_vec_b(t, :) = a1_b(:, 2: end); %1*400
%	a2_vec_b(t, :) = transpose(a2_b)(:, 2: end); %25*1
%	a3_vec_b(t, :) = transpose(a3_b); %1*10
%	delta3(:, t) = a3_b - transpose(y_matrix(t,:));  % 10*1
%	%disp("-----------");
%	%disp(size(transpose(y_matrix(t,:))));
%	%disp(size(transpose(Theta2(:,2:end)) * delta3));
%	%			       25*10, 				  delta3:10*1    25*1
%	delta2(:, t) = transpose(Theta2(:,2:end)) * delta3(:,t) .* sigmoidGradient(z2_b); 
%
%	%disp("------------------");
%	%disp(size(delta3));
%	%disp(delta3(:, t));
%end
a1_b = [ones(m, 1) X];  %5000*401
z2_b = a1_b * transpose(Theta1); %5000*25
a2_b = sigmoid(z2_b);
a2_b = [ones(m, 1) a2_b]; %5000*26
z3_b = a2_b * transpose(Theta2);
a3_b = sigmoid(z3_b); %5000*10
a1_vec_b = a1_b(:, 2:end);
a2_vec_b = a2_b(:, 2:end);
a3_vec_b = a3_b;
delta3 = a3_b - y_matrix; %5000*10
delta2 = delta3 * Theta2(:, 2:end) .* sigmoidGradient(z2_b);


a2_vec_b = [ones(size(a2_vec_b, 1), 1) a2_vec_b]; 
a1_vec_b = [ones(size(a1_vec_b, 1), 1) a1_vec_b];
DELTA2_UP = transpose(delta3) * a2_vec_b;
DELTA1_UP = transpose(delta2) * a1_vec_b;
Theta1_grad = 1/m * DELTA1_UP;
Theta2_grad = 1/m * DELTA2_UP;
%Theta1_grad = Theta1_grad(:, 2:end);
%Theta2_grad = Theta2_grad(:, 2:end);
%disp(DELTA2_UP);

% ----------------- bckprop end------------------

% ----------------- Regularized start------------------
RegularizedNNT1 = lambda/m * Theta1;
RegularizedNNT2 = lambda/m * Theta2;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + RegularizedNNT1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + RegularizedNNT2(:, 2:end);


% ----------------- Regularized end------------------

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
