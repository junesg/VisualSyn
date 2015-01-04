function net = update_net_siamese(net, grad, rate)

% Don't update adversarial params.
tmp = net.var_comp;

fields = fieldnames(net);
for f = 1:numel(fields),
    net.(fields{f}) = net.(fields{f}) - rate * grad.(fields{f});
end

net.var_comp = tmp;

end
