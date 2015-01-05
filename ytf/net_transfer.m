function recon = net_transfer(net, var, id, params)

Hid = bsxfun(@plus, net.var_to_hid1*var, net.bias_hid1);
Hid = Hid + net.id_to_hid1*id;
Hid = relu(Hid);

Hid = bsxfun(@plus, net.hid1_to_hid2*Hid, net.bias_hid2);
Hid = relu(Hid);

recon = bsxfun(@plus, net.hid2_to_vis*Hid, net.bias_vis);

end