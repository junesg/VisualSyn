function Var = net_infer_var(net, params, X)

Hid_var1 = bsxfun(@plus, net.vis_to_hid_var1*X, net.bias_hid_var1);
Hid_var1 = relu(Hid_var1);

Hid_var2 = bsxfun(@plus, net.hid_var1_to_hid_var2*Hid_var1, net.bias_hid_var2);
Hid_var2 = relu(Hid_var2);

Var = net.hid_var2_to_var*Hid_var2;
if strcmp(params.enc,'sigmoid')
    Var = sigmoid(Var);
elseif strcmp(params.enc,'relu'),
    Var = relu(Var);
end

end