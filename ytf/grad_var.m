function [ grad, cost, stats ] = grad_var(net, pars, X1, X2, X3, Y)

numdata = size(X1,2);
grad = struct;

X = { X1, X2, X3 };
HidId1 = cell(3,1);
HidId2 = cell(3,1);
Id = cell(3,1);
DeltaAdvId = cell(3,1);
HidVar1 = cell(3,1);
HidVar2 = cell(3,1);
Var = cell(3,1);
DeltaAdvVar = cell(3,1);
DeltaIdPred = cell(3,1);
HidRecon1 = cell(3,1);
HidRecon2 = cell(3,1);
Recon = cell(3,1);
DeltaRecon = cell(3,1);

%% Feed-forward ID.
for i = 1:3,
    HidId1{i} = bsxfun(@plus, net.vis_to_hid_id1*X{i}, net.bias_hid_id1);
    if ~pars.gradcheck,
        HidId1{i} = relu(HidId1{i});
    end
    HidId2{i} = bsxfun(@plus, net.hid_id1_to_hid_id2*HidId1{i}, net.bias_hid_id2);
    if ~pars.gradcheck,
        HidId2{i} = relu(HidId2{i});
    end
    Id{i} = net.hid_id2_to_id*HidId2{i};
    if strcmp(pars.enc,'sigmoid')
        Id{i} = sigmoid(Id{i});
    elseif strcmp(pars.enc,'relu'),
        Id{i} = relu(Id{i});
    end
end

%% Feed-forward Var.
for i = 1:3,
    HidVar1{i} = bsxfun(@plus, net.vis_to_hid_var1*X{i}, net.bias_hid_var1);
    if ~pars.gradcheck,
        HidVar1{i} = relu(HidVar1{i});
    end
    HidVar2{i} = bsxfun(@plus, net.hid_var1_to_hid_var2*HidVar1{i}, net.bias_hid_var2);
    if ~pars.gradcheck,
        HidVar2{i} = relu(HidVar2{i});
    end
    Var{i} = net.hid_var2_to_var*HidVar2{i};
    if strcmp(pars.enc,'sigmoid')
        Var{i} = sigmoid(Var{i});
    elseif strcmp(pars.enc,'relu'),
        Var{i} = relu(Var{i});
    end
end

%% Cost ID adv
cost_adv_id = 0;
small = 1e-9;
DeltaAdvId{1} = zeros(size(Id{1},1),numdata);
DeltaAdvId{2} = zeros(size(Id{2},1),numdata);
DeltaAdvId{3} = zeros(size(Id{3},1),numdata);
for i = 1:numdata,
    p12 = exp(Id{1}(:,i)'*net.id_comp*Id{2}(:,i));
    p23 = exp(Id{3}(:,i)'*net.id_comp*Id{2}(:,i));
    score = p12 / (p12 + p23);
    cost_adv_id = cost_adv_id + (Y(i)*log(score + small) + ...
                          (1-Y(i))*log(1 - score + small)) / numdata;
    DeltaAdvId{1}(:,i) = (Y(i)-score)*net.id_comp*Id{2}(:,i) / numdata;
    DeltaAdvId{3}(:,i) = (score-Y(i))*net.id_comp*Id{2}(:,i) / numdata;
    DeltaAdvId{2}(:,i) = ...
        (Y(i) - score)*net.id_comp'*(Id{1}(:,i) - Id{3}(:,i)) / numdata;
end

%% Cost ID prediction.
cost_id_pred = 0;
DeltaIdPred{1} = zeros(size(Id{1},1), numdata);
DeltaIdPred{2} = zeros(size(Id{2},1), numdata);
DeltaIdPred{3} = zeros(size(Id{3},1), numdata);
grad.id_pred = 0*net.id_pred;
stats.acc = 0;

for i = 1:numdata,
    score12 = Id{1}(:,i)'*net.id_pred*Id{2}(:,i);
    score12 = sigmoid(score12);
    score32 = Id{3}(:,i)'*net.id_pred*Id{2}(:,i);
    score32 = sigmoid(score32);
    cost_id_pred = cost_id_pred - (log(score12 + small) + log(1 - score32 + small)) / numdata;
    delta12 = -(1 - score12) / numdata;
    delta32 = score32 / numdata;
    grad.id_pred = grad.id_pred + delta12*Id{1}(:,i)*Id{2}(:,i)';
    grad.id_pred = grad.id_pred + delta32*Id{3}(:,i)*Id{2}(:,i)';
    DeltaIdPred{1}(:,i) = delta12*net.id_pred*Id{2}(:,i);
    DeltaIdPred{3}(:,i) = delta32*net.id_pred*Id{2}(:,i);
    DeltaIdPred{2}(:,i) = ...
        delta12*net.id_pred'*Id{1}(:,i) + delta32*net.id_pred'*Id{3}(:,i);

    stats.acc = stats.acc + (double(score12>0.5) + double(score32<0.5))/(2.0*numdata);
end

%% Cost Var adv.
cost_adv_var = 0;
DeltaAdvVar{1} = zeros(size(Var{1},1), numdata);
DeltaAdvVar{2} = zeros(size(Var{2},1), numdata);
DeltaAdvVar{3} = zeros(size(Var{3},1), numdata);
for i = 1:numdata,
    score12 = Var{1}(:,i)'*net.var_comp*Var{2}(:,i);
    score12 = sigmoid(score12);
    score32 = Var{3}(:,i)'*net.var_comp*Var{2}(:,i);
    score32 = sigmoid(score32);
    cost_adv_var = cost_adv_var + (log(score12 + small) + log(1 - score32 + small)) / numdata;
    delta12 = (1 - score12) / numdata;
    delta32 = -score32 / numdata;
    DeltaAdvVar{1}(:,i) = delta12*net.var_comp*Var{2}(:,i);
    DeltaAdvVar{3}(:,i) = delta32*net.var_comp*Var{2}(:,i);
    DeltaAdvVar{2}(:,i) = ...
        delta12*net.var_comp'*Var{1}(:,i) + delta32*net.var_comp'*Var{3}(:,i);
end

%% Cost reconstruction.
cost_recon = 0;
grad.hid1_to_hid2 = 0*net.hid1_to_hid2;
grad.hid2_to_vis = 0*net.hid2_to_vis;
grad.bias_hid1 = 0*net.bias_hid1;
grad.bias_hid2 = 0*net.bias_hid2;
grad.bias_vis = 0*net.bias_vis;
grad.var_to_hid1 = 0*net.var_to_hid1;
grad.id_to_hid1 = 0*net.id_to_hid1;
for i = 1:3,
    HidRecon1{i} = bsxfun(@plus, net.id_to_hid1*Id{i}, net.bias_hid1);
    HidRecon1{i} = HidRecon1{i} + net.var_to_hid1*Var{i};
    if ~pars.gradcheck,
        HidRecon1{i} = relu(HidRecon1{i});
    end
    HidRecon2{i} = bsxfun(@plus, net.hid1_to_hid2*HidRecon1{i}, net.bias_hid2);
    if ~pars.gradcheck,
        HidRecon2{i} = relu(HidRecon2{i});
    end
    Recon{i} = bsxfun(@plus, net.hid2_to_vis*HidRecon2{i}, net.bias_vis);
    err = Recon{i} - X{i};
    cost_recon = cost_recon + 0.5*pars.lambda*(err(:)'*err(:)) / numdata;
    DeltaRecon{i} = pars.lambda * err / numdata;
    grad.bias_vis = grad.bias_vis + sum(DeltaRecon{i},2);
    grad.hid2_to_vis = grad.hid2_to_vis + DeltaRecon{i}*HidRecon2{i}';
    DeltaRecon{i} = net.hid2_to_vis'*DeltaRecon{i};
    if ~pars.gradcheck,
        DeltaRecon{i} = DeltaRecon{i}.*(HidRecon2{i} > 0);
    end
    grad.bias_hid2 = grad.bias_hid2 + sum(DeltaRecon{i},2);
    grad.hid1_to_hid2 = grad.hid1_to_hid2 + DeltaRecon{i}*HidRecon1{i}';
    DeltaRecon{i} = net.hid1_to_hid2'*DeltaRecon{i};
    if ~pars.gradcheck,
        DeltaRecon{i} = DeltaRecon{i}.*(HidRecon1{i} > 0);
    end
    grad.id_to_hid1 = grad.id_to_hid1 + DeltaRecon{i}*Id{i}';
    grad.var_to_hid1 = grad.var_to_hid1 + DeltaRecon{i}*Var{i}';
end

%% Backprop all gradients.
DeltaId = cell(3,1);
DeltaVar = cell(3,1);
for i = 1:3,
    DeltaId{i} = DeltaAdvId{i} + DeltaIdPred{i} + net.id_to_hid1'*DeltaRecon{i};
    DeltaVar{i} = DeltaAdvVar{i} + net.var_to_hid1'*DeltaRecon{i};
    if strcmp(pars.enc,'sigmoid'),
        DeltaId{i} = DeltaId{i}.*(Id{i}.*(1-Id{i}));
    elseif strcmp(pars.enc, 'relu'),
        DeltaId{i} = DeltaId{i}.*(Id{i}>0);
    end
end

% ID pathway.
grad.hid_id2_to_id = 0*net.hid_id2_to_id;
grad.bias_hid_id2 = 0*net.bias_hid_id2;
grad.hid_id1_to_hid_id2 = 0*net.hid_id1_to_hid_id2;
grad.bias_hid_id1 = 0*net.bias_hid_id1;
grad.vis_to_hid_id1 = 0*net.vis_to_hid_id1;
for i = 1:3,    
    grad.hid_id2_to_id = grad.hid_id2_to_id + DeltaId{i} * HidId2{i}';
    DeltaId{i} = net.hid_id2_to_id'*DeltaId{i};
    if ~pars.gradcheck,
        DeltaId{i} = DeltaId{i}.*(HidId2{i} > 0);
    end
    grad.bias_hid_id2 = grad.bias_hid_id2 + sum(DeltaId{i},2);
    grad.hid_id1_to_hid_id2 = grad.hid_id1_to_hid_id2 + DeltaId{i}*HidId1{i}';
    DeltaId{i} = net.hid_id1_to_hid_id2'*DeltaId{i};
    if ~pars.gradcheck,
        DeltaId{i} = DeltaId{i}.*(HidId1{i} > 0);
    end
    grad.bias_hid_id1 = grad.bias_hid_id1 + sum(DeltaId{i},2);
    grad.vis_to_hid_id1 = grad.vis_to_hid_id1 + DeltaId{i}*X{i}';
end

% Var pathway.
grad.hid_var2_to_var = 0*net.hid_var2_to_var;
grad.bias_hid_var2 = 0*net.bias_hid_var2;
grad.hid_var1_to_hid_var2 = 0*net.hid_var1_to_hid_var2;
grad.bias_hid_var1 = 0*net.bias_hid_var1;
grad.vis_to_hid_var1 = 0*net.vis_to_hid_var1;
for i = 1:3,
    grad.hid_var2_to_var = grad.hid_var2_to_var + DeltaVar{i} * HidVar2{i}';
    DeltaVar{i} = net.hid_var2_to_var'*DeltaVar{i};
    if ~pars.gradcheck,
        DeltaVar{i} = DeltaVar{i}.*(HidVar2{i} > 0);
    end
    grad.bias_hid_var2 = grad.bias_hid_var2 + sum(DeltaVar{i},2);
    grad.hid_var1_to_hid_var2 = grad.hid_var1_to_hid_var2 + DeltaVar{i}*HidVar1{i}';
    DeltaVar{i} = net.hid_id1_to_hid_id2'*DeltaVar{i};
    if ~pars.gradcheck,
        DeltaVar{i} = DeltaVar{i}.*(HidVar1{i} > 0);
    end
    grad.bias_hid_var1 = grad.bias_hid_var1 + sum(DeltaVar{i},2);
    grad.vis_to_hid_var1 = grad.vis_to_hid_var1 + DeltaVar{i}*X{i}';
end

%% L2 regularization.
cost_l2r = 0.5*pars.l2reg*(net.vis_to_hid_id1(:)'*net.vis_to_hid_id1(:) + ...
                           net.hid_id1_to_hid_id2(:)'*net.hid_id1_to_hid_id2(:) + ...
                           net.vis_to_hid_var1(:)'*net.vis_to_hid_var1(:) + ...
                           net.hid_var1_to_hid_var2(:)'*net.hid_var1_to_hid_var2(:) + ...
                           net.hid_id2_to_id(:)'*net.hid_id2_to_id(:) + ...
                           net.hid_var2_to_var(:)'*net.hid_var2_to_var(:) + ...
                           net.id_to_hid1(:)'*net.id_to_hid1(:) + ...
                           net.id_pred(:)'*net.id_pred(:) + ...
                           net.var_to_hid1(:)'*net.var_to_hid1(:) + ...
                           net.hid1_to_hid2(:)'*net.hid1_to_hid2(:) + ...
                           net.hid2_to_vis(:)'*net.hid2_to_vis(:));

cost = cost_adv_id + cost_adv_var + cost_id_pred + cost_l2r + cost_recon;

grad.vis_to_hid_id1 = grad.vis_to_hid_id1 + pars.l2reg*net.vis_to_hid_id1;
grad.hid_id1_to_hid_id2 = grad.hid_id1_to_hid_id2 + pars.l2reg*net.hid_id1_to_hid_id2;
grad.vis_to_hid_var1 = grad.vis_to_hid_var1 + pars.l2reg*net.vis_to_hid_var1;
grad.hid_var1_to_hid_var2 = grad.hid_var1_to_hid_var2 + pars.l2reg*net.hid_var1_to_hid_var2;
grad.hid_id2_to_id = grad.hid_id2_to_id + pars.l2reg*net.hid_id2_to_id;
grad.hid_var2_to_var = grad.hid_var2_to_var + pars.l2reg*net.hid_var2_to_var;
grad.id_to_hid1 = grad.id_to_hid1 + pars.l2reg*net.id_to_hid1;
grad.var_to_hid1 = grad.var_to_hid1 + pars.l2reg*net.var_to_hid1;
grad.hid1_to_hid2 = grad.hid1_to_hid2 + pars.l2reg*net.hid1_to_hid2;
grad.hid2_to_vis = grad.hid2_to_vis + pars.l2reg*net.hid2_to_vis;
grad.id_pred = grad.id_pred + pars.l2reg*net.id_pred;

end
