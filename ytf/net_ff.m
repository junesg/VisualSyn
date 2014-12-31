function [ Hid_id, Hid_var, Id_pred, Var, Hid, Recon1, Recon2 ] = net_ff(net, X, Id, params)

Hid_id = bsxfun(@plus, net.vis_to_hid_id*X, net.bias_hid_id);
Hid_id = relu(Hid_id);

Hid_var = bsxfun(@plus, net.vis_to_hid_var*X, net.bias_hid_var);
Hid_var = relu(Hid_var);

Id_pred = softmax(net.hid_id_to_id, Hid_id);

Var = net.hid_var_to_var*Hid_var;
if strcmp(params.enc,'sigmoid')
    Var = sigmoid(Var);
elseif strcmp(params.enc,'relu'),
    Var = relu(Var);
end

Hid = bsxfun(@plus, net.id_to_hid*Id, net.bias_hid);
Hid = Hid + net.var_to_hid*Var;
Hid = relu(Hid);

Recon1 = bsxfun(@plus, net.hid_to_vis*Hid, net.bias_vis);

Hid2 = bsxfun(@plus, net.id_to_hid*Id_pred, net.bias_hid);
Hid2 = Hid2 + net.var_to_hid*Var;
Hid2 = relu(Hid2);

Recon2 = bsxfun(@plus, net.hid_to_vis*Hid2, net.bias_vis);

end