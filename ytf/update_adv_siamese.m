function net = update_adv_siamese(net, grad, rate)

net.var_comp = net.var_comp - rate * grad.var_comp;

end
