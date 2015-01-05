function [ grad, cost, stats ] = grad_adv_id(net, pars, X1, X2, X3)

numdata = size(X1,2);
grad = struct;

X = { X1, X2, X3 };
HidVar = cell(3,1);
Var = cell(3,1);

%% Feed-forward Var.
for i = 1:3,
    HidVar{i} = bsxfun(@plus, net.vis_to_hid_var1*X{i}, net.bias_hid_var1);
    if ~pars.gradcheck,
        HidVar{i} = relu(HidVar{i});
    end
    HidVar{i} = bsxfun(@plus, net.hid_var1_to_hid_var2*HidVar{i}, net.bias_hid_var2);
    if ~pars.gradcheck,
        HidVar{i} = relu(HidVar{i});
    end
    Var{i} = net.hid_var2_to_var*HidVar{i};
    if strcmp(pars.enc,'sigmoid')
        Var{i} = sigmoid(Var{i});
    elseif strcmp(pars.enc,'relu'),
        Var{i} = relu(Var{i});
    end
end

%% Cost adv.
cost_adv_var = 0;
small = 1e-9;
grad.var_comp = 0*net.var_comp;
stats.acc = 0;
for i = 1:numdata,
    score12 = Var{1}(:,i)'*net.var_comp*Var{2}(:,i);
    score12 = sigmoid(score12);
    score32 = Var{3}(:,i)'*net.var_comp*Var{2}(:,i);
    score32 = sigmoid(score32);
    cost_adv_var = cost_adv_var - (log(score12 + small) + log(1 - score32 + small)) / numdata;
    delta12 = -(1 - score12) / numdata;
    delta32 = score32 / numdata;
    grad.var_comp = grad.var_comp + ...
        delta12*Var{1}(:,i)*Var{2}(:,i)' + delta32*Var{3}(:,i)*Var{2}(:,i)';
    
    stats.acc = stats.acc + (double(score12>0.5) + double(score32<0.5))/(2.0*numdata);
end

%% L2 regularization.
cost_l2r = 0.5*pars.l2reg*(net.var_comp(:)'*net.var_comp(:));

cost = cost_adv_var + cost_l2r;

grad.var_comp = grad.var_comp + pars.l2reg*net.var_comp;

end
