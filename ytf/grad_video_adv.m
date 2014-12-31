function [ grad, cost ] = grad_video_adv(net, pars, X, Id)

numdata = size(X,2);
grad = struct;

Hid_var = bsxfun(@plus, net.vis_to_hid_var*X, net.bias_hid_var);
Hid_var = relu(Hid_var);

Hid_id = bsxfun(@plus, net.vis_to_hid_id*X, net.bias_hid_id);
Hid_id = relu(Hid_id);

Id_pred_id = softmax(net.hid_id_to_id, Hid_id);

Var = net.hid_var_to_var*Hid_var;
if strcmp(pars.enc,'sigmoid')
    Var = sigmoid(Var);
elseif strcmp(pars.enc,'relu'),
    Var = relu(Var);
end

Id_pred_var = softmax(net.var_to_id, Var);

Hid = bsxfun(@plus, net.id_to_hid*Id, net.bias_hid);
Hid = Hid + net.var_to_hid*Var;
Hid = relu(Hid);

Recon = bsxfun(@plus, net.hid_to_vis*Hid, net.bias_vis);

%% Backprop Reconstruction.
err_recon = Recon - X;
cost_recon = 0.5*err_recon(:)'*err_recon(:) / numdata;

delta_recon = err_recon / numdata;

grad.bias_vis = sum(delta_recon, 2);
grad.hid_to_vis = delta_recon*Hid';

delta_recon = net.hid_to_vis'*delta_recon;
delta_recon = delta_recon.*(Hid > 0);

grad.bias_hid = sum(delta_recon,2);
grad.id_to_hid = delta_recon*Id';
grad.var_to_hid = delta_recon*Var';

%% Add in ID prediction gradient from id.
small = 1e-9;
err_pred_id = -pars.lambda * (Id.*log(Id_pred_id + small));
cost_pred_id = sum(sum(err_pred_id)) / numdata;
delta_pred_id = -pars.lambda*(Id - Id_pred_id) / numdata;
delta_id = delta_pred_id;

%% Add in ID prediction gradient from var.
small = 1e-9;
err_pred_var = pars.lambda * (Id.*log(Id_pred_var + small));
cost_pred_var = sum(sum(err_pred_var)) / numdata;
delta_pred_var = pars.lambda*(Id - Id_pred_var) / numdata;

%% Continue backprop both gradients.
grad.hid_id_to_id = delta_id*Hid_id';

delta_id = net.hid_id_to_id'*delta_id;
delta_id = delta_id.*(Hid_id > 0);

grad.bias_hid_id = sum(delta_id,2);
grad.vis_to_hid_id = delta_id*X';

% Backprop var.
delta_var = net.var_to_hid'*delta_recon + net.var_to_id'*delta_pred_var;
if strcmp(pars.enc,'sigmoid')
    delta_var = delta_var.*(Var.*(1-Var));
elseif strcmp(pars.enc,'relu'),
    delta_var = delta_var.*(Var>0);
end

grad.hid_var_to_var = delta_var*Hid_var';
delta_var = net.hid_var_to_var'*delta_var;

delta_var = delta_var.*(Hid_var > 0);
grad.bias_hid_var = sum(delta_var,2);
grad.vis_to_hid_var = delta_var*X';

cost_l2r = 0.5*pars.l2reg*(net.vis_to_hid_id(:)'*net.vis_to_hid_id(:) + ...
                net.vis_to_hid_var(:)'*net.vis_to_hid_var(:) + ...
                net.hid_id_to_id(:)'*net.hid_id_to_id(:) + ...
                net.hid_var_to_var(:)'*net.hid_var_to_var(:) + ...
                net.id_to_hid(:)'*net.id_to_hid(:) + ...
                net.var_to_hid(:)'*net.var_to_hid(:) + ...
                net.hid_to_vis(:)'*net.hid_to_vis(:) + ...
                net.var_to_id(:)'*net.var_to_id(:));

cost = cost_pred_id + cost_pred_var + cost_recon + cost_l2r;

grad.var_to_id = pars.l2reg*net.var_to_id;
grad.vis_to_hid_id = grad.vis_to_hid_id + pars.l2reg*net.vis_to_hid_id;
grad.vis_to_hid_var = grad.vis_to_hid_var + pars.l2reg*net.vis_to_hid_var;
grad.hid_id_to_id = grad.hid_id_to_id + pars.l2reg*net.hid_id_to_id;
grad.hid_var_to_var = grad.hid_var_to_var + pars.l2reg*net.hid_var_to_var;
grad.id_to_hid = grad.id_to_hid + pars.l2reg*net.id_to_hid;
grad.var_to_hid = grad.var_to_hid + pars.l2reg*net.var_to_hid;
grad.hid_to_vis = grad.hid_to_vis + pars.l2reg*net.hid_to_vis;

end


