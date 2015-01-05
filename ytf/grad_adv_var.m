function [ grad, cost, stats ] = grad_adv_var(net, pars, X1, X2, X3, Y)

numdata = size(X1,2);
grad = struct;
Y = double(Y);

X = { X1, X2, X3 };
HidId = cell(3,1);
Id = cell(3,1);

for i = 1:3,
    HidId{i} = bsxfun(@plus, net.vis_to_hid_id1*X{i}, net.bias_hid_id1);
    if ~pars.gradcheck,
        HidId{i} = relu(HidId{i});
    end
    HidId{i} = bsxfun(@plus, net.hid_id1_to_hid_id2*HidId{i}, net.bias_hid_id2);
    if ~pars.gradcheck,
        HidId{i} = relu(HidId{i});
    end
    Id{i} = net.hid_id2_to_id*HidId{i};
    if strcmp(pars.enc,'sigmoid')
        Id{i} = sigmoid(Id{i});
    elseif strcmp(pars.enc,'relu'),
        Id{i} = relu(Id{i});
    end
end

cost_adv = 0;
small = 1e-9;
grad.id_comp = 0*net.id_comp;
stats.acc = 0;
for i = 1:numdata,
    p12 = exp(Id{1}(:,i)'*net.id_comp*Id{2}(:,i));
    p23 = exp(Id{3}(:,i)'*net.id_comp*Id{2}(:,i));
    score = p12 / (p12 + p23);    
    cost_adv = cost_adv - (Y(i)*log(score + small) + (1-Y(i))*log(1 - score + small));
    delta_idcomp = Y(i)*(Id{1}(:,i)*Id{2}(:,i)' - ...
                         score*Id{1}(:,i)*Id{2}(:,i)' - ...
                         (1-score)*Id{3}(:,i)*Id{2}(:,i)') + ...
                (1-Y(i))*(Id{3}(:,i)*Id{2}(:,i)' - ...
                         score*Id{1}(:,i)*Id{2}(:,i)' - ...
                         (1-score)*Id{3}(:,i)*Id{2}(:,i)');
    grad.id_comp = grad.id_comp - delta_idcomp;

    stats.acc = stats.acc + (Y(i)*double(score>0.5) + (1-Y(i))*double(score<=0.5)) / numdata;
end
cost_adv = cost_adv / numdata;
grad.id_comp = grad.id_comp / numdata;

cost = cost_adv + 0.5*pars.l2reg*(net.id_comp(:)'*net.id_comp(:));

grad.id_comp = grad.id_comp + pars.l2reg*net.id_comp;

end


