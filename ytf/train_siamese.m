function [ net, params ] = train_siamese(X, Y, params)

% Make all pairs.
pairs = makepairs(Y);
numpair = size(pairs,1);

net = init_net_siamese(params, 0.01);
rate = params.epsilon;
for i = 1:params.maxiter,
  % Sample 2000 pairs for this epoch.
  R = randsample(numpair, params.numsample);
  P = pairs(R,:);
  numbatch = floor(params.numsample / params.batchsize);
  for b = 1:numbatch,
    batchidx = (1+(b-1)*params.batchsize):(b*params.batchsize);
    Pb = P(batchidx,:);
    Xbref = X(:,Pb(:,1)); % Ref
    Xbmatch = X(:,Pb(:,2)); % Match
    Xbnomatch = X(:,Pb(:,3)); % Nonmatch
    Xb1 = [ Xbref, Xbref ];
    Xb2 = [ Xbmatch, Xbnomatch ];
    Yb = [ ones(params.batchsize,1) ; zeros(params.batchsize,1) ];

    % Update adversary.
    for s = 1:3,
        [ grad, cost_adv ] = grad_adv_siamese(net, params, Xb1, Xb2, Yb);
    end
    net = update_adv_siamese(net, grad, rate);

    % Update network.
    [ grad, cost ] = grad_video_siamese(net, params, Xb1, Xb2, Yb);
    net = update_net_siamese(net, grad, rate);

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

