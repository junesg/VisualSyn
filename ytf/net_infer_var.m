function Var = net_infer_var(net, params, X)

Hid_id = bsxfun(@plus, net.vis_to_hid_id*X, net.bias_hid_id);
Hid_id = relu(Hid_id);

Hid_var = bsxfun(@plus, net.vis_to_hid_var*X, net.bias_hid_var);
Hid_var = relu(Hid_var);

Var = net.hid_var_to_var*Hid_var;
if strcmp(params.enc,'sigmoid')
    Var = sigmoid(Var);
elseif strcmp(params.enc,'relu'),
    Var = relu(Var);
end

end