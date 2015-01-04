function [ Id1, Var1, Recon ] = net_ff_siamese(net, X1, pars)

% Var inference.
Hid_var1 = bsxfun(@plus, net.vis_to_hid_var*X1, net.bias_hid_var);
Hid_var1 = relu(Hid_var1);
Var1 = net.hid_var_to_var*Hid_var1;
if strcmp(pars.enc,'sigmoid')
    Var1 = sigmoid(Var1);
elseif strcmp(pars.enc,'relu'),
    Var1 = relu(Var1);
end

% ID inference.
Hid_id1 = bsxfun(@plus, net.vis_to_hid_id*X1, net.bias_hid_id);
Hid_id1 = relu(Hid_id1);
Id1 = net.hid_id_to_id*Hid_id1;
if strcmp(pars.enc,'sigmoid')
    Id1 = sigmoid(Id1);
elseif strcmp(pars.enc,'relu'),
    Id1 = relu(Id1);
end

%% Reconstruction inference.
Hid = bsxfun(@plus, net.id_to_hid*Id1, net.bias_hid);
Hid = Hid + net.var_to_hid*Var1;
Hid = relu(Hid);

Recon = bsxfun(@plus, net.hid_to_vis*Hid, net.bias_vis);

end
