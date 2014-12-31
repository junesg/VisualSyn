function [ grad, cost ] = grad_adv_siamese(net, pars, X1, X2, Y)

numdata = size(X1,2);
grad = struct;

Hid_var1 = bsxfun(@plus, net.vis_to_hid_var*X1, net.bias_hid_var);
Hid_var1 = relu(Hid_var1);
Var1 = net.hid_var_to_var*Hid_var1;
if strcmp(pars.enc,'sigmoid')
    Var1 = sigmoid(Var1);
elseif strcmp(pars.enc,'relu'),
    Var1 = relu(Var1);
end

Hid_var2 = bsxfun(@plus, net.vis_to_hid_var*X2, net.bias_hid_var);
Hid_var2 = relu(Hid_var2);
Var2 = net.hid_var_to_var*Hid_var2;
if strcmp(pars.enc,'sigmoid')
    Var2 = sigmoid(Var2);
elseif strcmp(pars.enc,'relu'),
    Var2 = relu(Var2);
end

cost_adv = 0;
small = 1e-9;
grad.var_comp = 0*net.var_comp;
for i = 1:numdata,
    score = Var1(:,i)'*net.var_comp*Var2(:,i);
    score = sigmoid(score);
    cost_adv = cost_adv - (Y(i)*log(score + small) + (1-Y(i))*log(1 - score + small));
    delta = -(Y(i) - score);
    grad.var_comp = grad.var_comp + delta*Var1(:,i)*Var2(:,i)';
end

cost = cost_adv + 0.5*pars.l2reg*(net.var_comp(:)'*net.var_comp(:));

grad.var_comp = grad.var_comp + pars.l2reg*net.var_comp;

end


