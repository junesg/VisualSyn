function [ weights, params ] = train_video(X, ids, params)

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
        sample_idx = randsample(size(X{u},2), numframe, 1);
        Xepoch(:,1+(u-1)*numframe:u*numframe) = X{u}(:,sample_idx);
        Yepoch(u,1+(u-1)*numframe:u*numframe) = 1;
    end

    numbatch = floor(size(Xepoch,2)/params.batchsize);
    for b = 1:numbatch,
        Xb = Xepoch(:,1+(b-1)*batchsize:b*batchsize);
        Yb = Yepoch(:,1+(b-1)*batchsize:b*batchsize);

        [ grad, cost ] = grad_video(weights,params,Xb,Yb);
        weights = update_net(weights, grad, rate);

        % For visualization.
        if mod(b,10)==0,
            fprintf(1,'\tbatch %d of %d cost=%.4f\n', ...
                b, numbatch, cost);
            [ ~, ~, ~, ~, ~, Rb1, Rb2 ] = net_ff(weights, Xb, Yb, params);
            subplot(1,3,1);
            display_network_nonsquare(Xb(:,1:16));
            subplot(1,3,2);
            display_network_nonsquare(Rb1(:,1:16));
            subplot(1,3,3);
            display_network_nonsquare(Rb2(:,1:16));
        end
    end

    rate = lr*(exp_decay^(i/decay_epoch));
    % Checkpoint.
    save(params.fname_mat,'weights','params');
end

save(params.fname_mat,'weights','params');

end