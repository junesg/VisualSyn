function [ net, params ] = train_var_semisup(X, Y, params)

% Make all pairs.
pairs = makepairs(Y);
numpair = size(pairs,1);
numstep = 1;

net = init_net_var(params, 0.01);
rate = params.epsilon;
for i = 1:params.maxiter,
  % Sample 2000 pairs for this epoch.
  R = randsample(numpair, params.numsample);
  P = pairs(R,:);
  numbatch = floor(params.numsample / params.batchsize);
  for b = 1:numbatch,
    batchidx = (1+(b-1)*params.batchsize):(b*params.batchsize);
    Pb = P(batchidx,:);
    Xb1 = X(:,Pb(:,1)); % Ref
    Xb2 = X(:,Pb(:,2)); % Match
    Xb3 = X(:,Pb(:,3)); % Nonmatch

    % Find indices where || Xb1 - Xb2 || < || Xb3 - Xb2 ||
    Var1 = net_infer_var(net, params, Xb1);
    Var2 = net_infer_var(net, params, Xb2);
    Var3 = net_infer_var(net, params, Xb3);
    score12 = sum((Var1-Var2).^2,1);
    score32 = sum((Var3-Var2).^2,1);
    Yb = score12 < score32;

    % Update adversary.
    for s = 1:numstep,
        [ gvar, cost_adv_var ] = grad_adv_id(net, params, Xb1, Xb2, Xb3);        
        if rand() < 0.5,
            [ gid, cost_adv_id ] = grad_adv_var(net,params,Xb1,Xb2,Xb3,Yb);
        else
            [ gid, cost_adv_id ] = grad_adv_var(net,params,Xb3,Xb2,Xb1,~Yb);            
        end
        cost_adv = cost_adv_var + cost_adv_id;
        grad = gid;
        grad.var_comp = gvar.var_comp;
        net = update_adv_var(net, grad, rate);
    end

    % Update network.
    [ grad, cost ] = grad_var(net, params, Xb1, Xb2, Xb3, Yb);
    net = update_net_var(net, grad, rate);

    if mod(b,40)==0,
      fprintf(1,'Epoch %d, batch %d of %d, rate=%.4f, cost_adv=%.4f, cost=%.4f\n', ...
              i, b, numbatch, rate, cost_adv, cost);  
      [ ~, Var1, Recon1 ] = net_ff_siamese(net, Xb1, params);
      [ Id2, ~, Recon2 ] = net_ff_siamese(net, Xb2, params);
      Transfer = net_transfer(net, Var1, Id2, params);
      subplot(2,2,1);
      num = min(size(Xb1,2),4);
      display_network_nonsquare(Xb1(:,1:num));
      subplot(2,2,2);
      display_network_nonsquare(Recon1(:,1:num));
      subplot(2,2,3);
      display_network_nonsquare(Recon2(:,1:num));
      subplot(2,2,4);
      display_network_nonsquare(Transfer(:,1:num));
    end
  end

  rate = params.epsilon*(params.eps_decay)^(i/params.decay_epoch);
end

end
