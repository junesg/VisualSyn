function weights = init_net_var(params, stdinit)

weights = struct;

weights.vis_to_hid_id1 = stdinit*randn(params.numhid_id1,params.numvis);
weights.hid_id1_to_hid_id2 = stdinit*randn(params.numhid_id2,params.numhid_id1);

weights.vis_to_hid_var1 = stdinit*randn(params.numhid_var1,params.numvis);
weights.hid_var1_to_hid_var2 = stdinit*randn(params.numhid_var2,params.numhid_var1);

weights.hid_id2_to_id = stdinit*randn(params.numid,params.numhid_id2);
weights.hid_var2_to_var = stdinit*randn(params.numvar,params.numhid_var2);

weights.id_to_hid1 = stdinit*randn(params.numhid1,params.numid);
weights.var_to_hid1 = stdinit*randn(params.numhid1,params.numvar);
weights.hid1_to_hid2 = stdinit*randn(params.numhid2,params.numhid1);

weights.var_comp = stdinit*randn(params.numvar,params.numvar);
weights.id_comp = stdinit*randn(params.numid,params.numid);
weights.id_pred = stdinit*randn(params.numid,params.numid);

weights.hid2_to_vis = stdinit*randn(params.numvis,params.numhid2);

weights.bias_hid_id1 = zeros(params.numhid_id1,1);
weights.bias_hid_id2 = zeros(params.numhid_id2,1);

weights.bias_hid_var1 = zeros(params.numhid_var1,1);
weights.bias_hid_var2 = zeros(params.numhid_var2,1);

weights.bias_hid1 = zeros(params.numhid1,1);
weights.bias_hid2 = zeros(params.numhid2,1);
weights.bias_vis = zeros(params.numvis,1);

end