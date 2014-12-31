function net = update_net(net, grad, rate)

% Don't update adversarial params.
tmp = net.var_to_id;

fields = fieldnames(net);
for f = 1:numel(fields),
    net.(fields{f}) = net.(fields{f}) - rate * grad.(fields{f});
end

net.var_to_id = tmp;

end