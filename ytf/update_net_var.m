function net = update_net_var(net, grad, rate)

% Don't update adversarial params.
tmp1 = net.var_comp;
tmp2 = net.id_comp;

fields = fieldnames(grad);
for f = 1:numel(fields),
    net.(fields{f}) = net.(fields{f}) - rate * grad.(fields{f});
    %net.(fields{f}) = net.(fields{f}) + rate * grad.(fields{f});
end

net.var_comp = tmp1;
net.id_comp = tmp2;

end
