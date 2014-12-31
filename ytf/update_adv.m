function net = update_adv(net, grad, rate)

net.var_to_id = net.var_to_id - rate * grad.var_to_id;

end