function [ weights, params ] = train_video_adv(X, ids, params)

stdinit = 0.01;
weights = init_net(params, stdinit);
% Train model
lr = params.epsilon;
maxiter = params.maxiter;
decay_epoch = 50;
exp_decay = 0.8;

uids = unique(ids);
numframe = 20;
batchsize = params.batchsize;
rate = lr;

for i = 1:maxiter,
    fprintf(1,'epoch %d of %d rate = %f\n', i, maxiter, rate);

    Xepoch = zeros(size(X{1},1),length(uids)*numframe);
    Yepoch = zeros(length(uids),length(uids)*numframe);
    for u = 1:length(uids),
        movidx = find(ids==uids(u));
        movidx = randsample(movidx,1);
        sample_idx = randsample(size(X{movidx},2), numframe, 1);
        Xepoch(:,1+(u-1)*numframe:u*numframe) = X{movidx}(:,sample_idx);
        Yepoch(u,1+(u-1)*numframe:u*numframe) = 1;
    end

    numbatch = floor(size(Xepoch,2)/params.batchsize);
    for b = 1:numbatch,
        Xb = Xepoch(:,1+(b-1)*batchsize:b*batchsize);
        Yb = Yepoch(:,1+(b-1)*batchsize:b*batchsize);

        % Update adversarial network.
        for j=1:3,
            [ grad, cost_adv ] = grad_adv(weights,params,Xb,Yb);
            weights = update_adv(weights, grad, 1*rate);
            %{
            if mod(b,10)==0,
                fprintf(1,'\t\tcost_adv = %.8f\n', cost_adv);
            end
            %}
        end
        
        % Update inference network.
        [ grad, cost ] = grad_video_adv(weights,params,Xb,Yb);
        weights = update_net(weights, grad, rate);

        % For visualization.
        if mod(b,100)==0,
            fprintf(1,'\tbatch %d of %d cost=%.4f, adv=%.4f\n', ...
                b, numbatch, cost, cost_adv);
            [ ~, ~, ~, ~, ~, Rb1, Rb2 ] = net_ff(weights, Xb, Yb, params);
            subplot(1,3,1);
            num = min(size(Xb,2),16);
            display_network_nonsquare(Xb(:,1:num));
            subplot(1,3,2);
            display_network_nonsquare(Rb1(:,1:num));
            subplot(1,3,3);
            display_network_nonsquare(Rb2(:,1:num));
        end
    end

    rate = lr*(exp_decay^(i/decay_epoch));
    % Checkpoint.
    save(params.fname_mat,'weights','params');
end

save(params.fname_mat,'weights','params');

end