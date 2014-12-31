function [ grad, cost ] = grad_video_siamese(net, pars, X1, X2, Y)

numdata = size(X1,2);
grad = struct;

% Var inference.
Hid_var1 = bsxfun(@plus, net.vis_to_hid_var*X1, net.bias_hid_var);
Hid_var1 = relu(Hid_var1);
Var1 = net.hid_var_to_var*Hid_var1;
if strcmp(pars.enc,'sigmoid')
    Var1 = sigmoid(Var1);
elseif strcmp(pars.enc,'relu'),
    Var1 = relu(Var1);
end

Hid_var2 = bsxfun(@plus, net.vis_to_hid_var*X2, net.bias_hid_var);
Hid_var2 = relu(Hid_var2);
Var2 = net.hid_var_to_var*Hid_var2;
if strcmp(pars.enc,'sigmoid')
    Var2 = sigmoid(Var2);
elseif strcmp(pars.enc,'relu'),
    Var2 = relu(Var2);
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

Hid_id2 = bsxfun(@plus, net.vis_to_hid_id*X2, net.bias_hid_id);
Hid_id2 = relu(Hid_id2);
Id2 = net.hid_id_to_id*Hid_id2;
if strcmp(pars.enc,'sigmoid')
    Id2 = sigmoid(Id2);
elseif strcmp(pars.enc,'relu'),
    Id2 = relu(Id2);
end

%% Reconstruction inference.
Hid = bsxfun(@plus, net.id_to_hid*Id1, net.bias_hid);
Hid = Hid + net.var_to_hid*Var1;
Hid = relu(Hid);

Recon = bsxfun(@plus, net.hid_to_vis*Hid, net.bias_vis);

%% Id inference from Id.
cost_pred_id = 0;
small = 1e-9;
grad.id_comp = 0*net.id_comp;
delta_pred_id1 = zeros(size(Id1,1),numdata);
delta_pred_id2 = zeros(size(Id2,1),numdata);
for i = 1:numdata,
    score = Id1(:,i)'*net.id_comp*Id2(:,i);
    score = sigmoid(score);
    cost_pred_id = cost_pred_id - (Y(i)*log(score + small) + (1-Y(i))*log(1 - score + small));
    delta = -(Y(i) - score);
    grad.id_comp = grad.id_comp + delta*Id1(:,i)*Id2(:,i)';
    delta_pred_id1(:,i) = delta*net.id_comp*Id2(:,i);
    delta_pred_id2(:,i) = delta*(Id1(:,i)'*net.id_comp)';
end
cost_pred_id = cost_pred_id / numdata;
grad.id_comp = grad.id_comp / numdata;
delta_pred_id1 = delta_pred_id1 / numdata;
delta_pred_id2 = delta_pred_id2 / numdata;

%% Id inference from Var.
cost_pred_var = 0;
grad.var_comp = 0*net.var_comp;
delta_pred_var1 = zeros(size(Var1,1),numdata);
delta_pred_var2 = zeros(size(Var2,1),numdata);
for i = 1:numdata,
    score = Var1(:,i)'*net.var_comp*Var2(:,i);
    score = sigmoid(score);
    cost_pred_var = cost_pred_var + (Y(i)*log(score + small) + (1-Y(i))*log(1 - score + small));
    delta = (Y(i) - score);
    delta_pred_var1(:,i) = delta*net.var_comp*Var2(:,i);
    delta_pred_var2(:,i) = delta*(Var1(:,i)'*net.var_comp)';
end
cost_pred_var = cost_pred_var / numdata;
delta_pred_var1 = delta_pred_var1 / numdata;
delta_pred_var2 = delta_pred_var2 / numdata;

%% Backprop Reconstruction.
err_recon = Recon - X1;
cost_recon = pars.lambda*0.5*err_recon(:)'*err_recon(:) / numdata;

delta_recon = pars.lambda * err_recon / numdata;

grad.bias_vis = sum(delta_recon, 2);
grad.hid_to_vis = delta_recon*Hid';

delta_recon = net.hid_to_vis'*delta_recon;
delta_recon = delta_recon.*(Hid > 0);

grad.bias_hid = sum(delta_recon,2);
grad.id_to_hid = delta_recon*Id1';
grad.var_to_hid = delta_recon*Var1';

%% Add in ID prediction gradient from id1.
delta = net.id_to_hid'*delta_recon;
delta = delta + delta_pred_id1;
if strcmp(pars.enc,'sigmoid'),
    delta = delta.*(Id1.*(1-Id1));
elseif strcmp(pars.enc,'relu'),
    delta = delta.*(Id1 > 0);
end
grad.hid_id_to_id = delta*Hid_id1';
delta = net.hid_id_to_id'*delta;
delta = delta.*(Hid_id1 > 0);
grad.bias_hid_id = sum(delta,2);
grad.vis_to_hid_id = delta*X1';

%% Add in ID prediction gradient from id2.
delta = delta_pred_id2;
if strcmp(pars.enc,'sigmoid'),
    delta = delta.*(Id2.*(1-Id2));
elseif strcmp(pars.enc,'relu'),
    delta = delta.*(Id2 > 0);
end
grad.hid_id_to_id = grad.hid_id_to_id + delta*Hid_id2';
delta = net.hid_id_to_id'*delta;
delta = delta.*(Hid_id2 > 0);
grad.bias_hid_id = grad.bias_hid_id + sum(delta,2);
grad.vis_to_hid_id = grad.vis_to_hid_id + delta*X2';

%% Add in ID prediction gradient from var1.
delta = net.var_to_hid'*delta_recon;
delta = delta + delta_pred_var1;
if strcmp(pars.enc,'sigmoid'),
    delta = delta.*(Var1.*(1-Var1));
elseif strcmp(pars.enc,'relu'),
    delta = delta.*(Var1 > 0);
end
grad.hid_var_to_var = delta*Hid_var1';
delta = net.hid_var_to_var'*delta;
delta = delta.*(Hid_var1 > 0);
grad.bias_hid_var = sum(delta,2);
grad.vis_to_hid_var = delta*X1';

%% Add in ID prediction gradient from var2.
delta = delta_pred_var2;
if strcmp(pars.enc,'sigmoid'),
    delta = delta.*(Var2.*(1-Var2));
elseif strcmp(pars.enc,'relu'),
    delta = delta.*(Var2 > 0);
end
grad.hid_var_to_var = grad.hid_var_to_var + delta*Hid_var2';
delta = net.hid_var_to_var'*delta;
delta = delta.*(Hid_var2 > 0);
grad.bias_hid_var = grad.bias_hid_var + sum(delta,2);
grad.vis_to_hid_var = grad.vis_to_hid_var + delta*X2';

%% L2 reg

cost_l2r = 0.5*pars.l2reg*(net.vis_to_hid_id(:)'*net.vis_to_hid_id(:) + ...
                           net.vis_to_hid_var(:)'*net.vis_to_hid_var(:) + ...
                           net.hid_id_to_id(:)'*net.hid_id_to_id(:) + ...
                           net.hid_var_to_var(:)'*net.hid_var_to_var(:) + ...
                           net.id_to_hid(:)'*net.id_to_hid(:) + ...
                           net.var_to_hid(:)'*net.var_to_hid(:) + ...
                           net.hid_to_vis(:)'*net.hid_to_vis(:) + ...
                           net.id_comp(:)'*net.id_comp(:));

cost = cost_pred_id + cost_pred_var + cost_recon + cost_l2r;

grad.vis_to_hid_id = grad.vis_to_hid_id + pars.l2reg*net.vis_to_hid_id;
grad.vis_to_hid_var = grad.vis_to_hid_var + pars.l2reg*net.vis_to_hid_var;
grad.hid_id_to_id = grad.hid_id_to_id + pars.l2reg*net.hid_id_to_id;
grad.hid_var_to_var = grad.hid_var_to_var + pars.l2reg*net.hid_var_to_var;
grad.id_to_hid = grad.id_to_hid + pars.l2reg*net.id_to_hid;
grad.var_to_hid = grad.var_to_hid + pars.l2reg*net.var_to_hid;
grad.hid_to_vis = grad.hid_to_vis + pars.l2reg*net.hid_to_vis;
grad.id_comp = grad.id_comp + pars.l2reg*net.id_comp;

end


