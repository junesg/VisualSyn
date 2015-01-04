function net = update_adv_var(net, grad, rate)

net.var_comp = net.var_comp - rate * grad.var_comp;
net.id_comp = net.id_comp - rate * grad.id_comp;

end
