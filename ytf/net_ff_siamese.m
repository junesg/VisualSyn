function [ Id, Var, Recon ] = net_ff_siamese(net, X, pars)

HidId1 = bsxfun(@plus, net.vis_to_hid_id1*X, net.bias_hid_id1);
if ~pars.gradcheck,
    HidId1 = relu(HidId1);
end
HidId2 = bsxfun(@plus, net.hid_id1_to_hid_id2*HidId1, net.bias_hid_id2);
if ~pars.gradcheck,
    HidId2 = relu(HidId2);
end
Id = net.hid_id2_to_id*HidId2;
if strcmp(pars.enc,'sigmoid')
    Id = sigmoid(Id);
elseif strcmp(pars.enc,'relu'),
    Id = relu(Id);
end

HidVar1 = bsxfun(@plus, net.vis_to_hid_var1*X, net.bias_hid_var1);
if ~pars.gradcheck,
    HidVar1 = relu(HidVar1);
end
HidVar2 = bsxfun(@plus, net.hid_var1_to_hid_var2*HidVar1, net.bias_hid_var2);
if ~pars.gradcheck,
    HidVar2 = relu(HidVar2);
end
Var = net.hid_var2_to_var*HidVar2;
if strcmp(pars.enc,'sigmoid')
    Var = sigmoid(Var);
elseif strcmp(pars.enc,'relu'),
    Var = relu(Var);
end

Hid1 = bsxfun(@plus, net.id_to_hid1*Id, net.bias_hid1);
Hid1 = Hid1 + net.var_to_hid1*Var;
Hid1 = relu(Hid1);

Hid = bsxfun(@plus, net.hid1_to_hid2*Hid1, net.bias_hid2);
Hid = relu(Hid);

Recon = bsxfun(@plus, net.hid2_to_vis*Hid, net.bias_vis);

end
