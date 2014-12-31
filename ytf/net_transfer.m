function recon = net_transfer(net, var, id, params)

Hid = bsxfun(@plus, net.var_to_hid*var, net.bias_hid);
Hid = Hid + net.id_to_hid*id;
Hid = relu(Hid);

recon = bsxfun(@plus, net.hid_to_vis*Hid, net.bias_vis);

end