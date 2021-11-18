function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


a1=[ones(m,1) X]';
z2=Theta1*a1;
a2=[ones(1,m);sigmoid(z2)];
z3=Theta2*a2;
a3=sigmoid(z3);
[~,ymesh]=meshgrid(1:m,1:numel(unique(y)));
y=double((repmat(y',numel(unique(y)),1)==ymesh));
J=a3;
J(y==1)=log(a3(y==1));
J(y==0)=log(1-a3(y==0));
J=(-1/m)*sum(J(:))+(lambda/2/m)*((sum(sum(Theta1(:,2:end).^2)))+(sum(sum(Theta2(:,2:end).^2))));

delta3=a3-y;
delta2=Theta2'*delta3;
delta2=delta2(2:end,:).*sigmoidGradient(z2);


Delta1=delta2*a1';
Delta2=delta3*a2';

Theta1_grad=(1/m)*Delta1 + (lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad=(1/m)*Delta2 + (lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,2:end)];



grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
