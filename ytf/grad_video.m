function [ grad, cost ] = grad_video(net, pars, X, Id)

numdata = size(X,2);
grad = struct;

Hid_id = bsxfun(@plus, net.vis_to_hid_id*X, net.bias_hid_id);
Hid_id = relu(Hid_id);

Hid_var = bsxfun(@plus, net.vis_to_hid_var*X, net.bias_hid_var);
Hid_var = relu(Hid_var);

Id_pred = softmax(net, Hid_id);

Var = net.hid_var_to_var*Hid_var;
Var = relu(Var);

Hid = bsxfun(@plus, net.id_to_hid*Id, net.bias_hid);
Hid = Hid + net.var_to_hid*Var;
Hid = relu(Hid);

Recon = bsxfun(@plus, net.hid_to_vis*Hid, net.bias_vis);

%% Backprop Reconstruction.
err_recon = Recon - X;
cost_recon = 0.5*err_recon(:)'*err_recon(:) / numdata;

delta = err_recon / numdata;

grad.bias_vis = sum(delta, 2);
grad.hid_to_vis = delta*Hid';

delta = net.hid_to_vis'*delta;
delta = delta.*(Hid > 0);

grad.bias_hid = sum(delta,2);
%grad.id_to_hid = delta*Id_pred';
grad.id_to_hid = delta*Id';
grad.var_to_hid = delta*Var';

%{
delta_id = net.id_to_hid'*delta;

% backprop through softmax.
delta_tmp = 0*delta_id;
for i = 1:pars.numid,
    delta_tmp(i,:) = delta_tmp(i,:) + ...
        delta_id(i,:).*Id_pred(i,:).*(1-Id_pred(i,:));
    for j = 1:pars.numid,
        if (i==j),
            continue;
        end
        delta_tmp(i,:) =  delta_tmp(i,:) - ...
            delta_id(j,:).*(Id_pred(i,:).*Id_pred(j,:));
    end
end
delta_id = delta_tmp;
%}

%% Add in ID prediction gradient.
small = 1e-9;
err_pred = -pars.lambda * (Id.*log(Id_pred + small));
cost_pred = sum(sum(err_pred)) / numdata;
delta_pred = -pars.lambda*(Id - Id_pred) / numdata;

delta_id = delta_pred;

%% Continue backprop both gradients.
grad.hid_id_to_id = delta_id*Hid_id';

delta_id = net.hid_id_to_id'*delta_id;
delta_id = delta_id.*(Hid_id > 0);

grad.bias_hid_id = sum(delta_id,2);
grad.vis_to_hid_id = delta_id*X';

% Backprop var.
delta_var = net.var_to_hid'*delta;
delta_var = delta_var.*(Var > 0);

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
                net.hid_to_vis(:)'*net.hid_to_vis(:));

cost = cost_pred + cost_recon + cost_l2r;

grad.vis_to_hid_id = grad.vis_to_hid_id + pars.l2reg*net.vis_to_hid_id;
grad.vis_to_hid_var = grad.vis_to_hid_var + pars.l2reg*net.vis_to_hid_var;
grad.hid_id_to_id = grad.hid_id_to_id + pars.l2reg*net.hid_id_to_id;
grad.hid_var_to_var = grad.hid_var_to_var + pars.l2reg*net.hid_var_to_var;
grad.id_to_hid = grad.id_to_hid + pars.l2reg*net.id_to_hid;
grad.var_to_hid = grad.var_to_hid + pars.l2reg*net.var_to_hid;
grad.hid_to_vis = grad.hid_to_vis + pars.l2reg*net.hid_to_vis;

end

function P = softmax(net, H)

D = net.hid_id_to_id*H;
Dpre = bsxfun(@minus, D, max(D, [], 1));
D = exp(Dpre);
Dsum = sum(D);
P = bsxfun(@rdivide, D, Dsum);

end

