function [ grad, cost ] = grad_adv(net, pars, X, Id)

numdata = size(X,2);
grad = struct;

Hid_var = bsxfun(@plus, net.vis_to_hid_var*X, net.bias_hid_var);
Hid_var = relu(Hid_var);

Var = net.hid_var_to_var*Hid_var;
if strcmp(pars.enc,'sigmoid')
    Var = sigmoid(Var);
elseif strcmp(pars.enc,'relu'),
    Var = relu(Var);
end
Id_pred = softmax(net.var_to_id, Var);

%% Add in ID prediction gradient.
small = 1e-9;
err_pred = pars.lambda * (Id.*log(Id_pred + small));
cost = sum(sum(err_pred)) / numdata;
delta = pars.lambda*(Id - Id_pred) / numdata;

cost = cost + 0.5*pars.l2reg*(net.var_to_id(:)'*net.var_to_id(:));

%% Backprop var.
grad.var_to_id = delta*Var' + pars.l2reg*net.var_to_id;

end


