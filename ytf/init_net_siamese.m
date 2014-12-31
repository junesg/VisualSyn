function weights = init_net_siamese(params, stdinit)

weights = struct;
weights.vis_to_hid_id = stdinit*randn(params.numhid_id,params.numvis);
weights.vis_to_hid_var = stdinit*randn(params.numhid_var,params.numvis);
weights.hid_id_to_id = stdinit*randn(params.numid,params.numhid_id);
weights.hid_var_to_var = stdinit*randn(params.numvar,params.numhid_var);
weights.id_to_hid = stdinit*randn(params.numhid,params.numid);
weights.var_to_hid = stdinit*randn(params.numhid,params.numvar);
weights.var_comp = stdinit*randn(params.numvar,params.numvar);
weights.id_comp = stdinit*randn(params.numid,params.numid);
weights.hid_to_vis = stdinit*randn(params.numvis,params.numhid);
weights.bias_hid_id = zeros(params.numhid_id,1);
weights.bias_hid_var = zeros(params.numhid_var,1);
weights.bias_hid = zeros(params.numhid,1);
weights.bias_vis = zeros(params.numvis,1);

end