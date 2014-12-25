function [ weights, params ] = train_video(X, ids, params)

stdinit = 0.01;
weights = init_net(params, stdinit);
% Train model
lr = params.epsilon;
maxiter = params.maxiter;

uids = unique(ids);
K = 3;
skip = 2;

for i = 1:maxiter,
    % At each iteration, sample K consecutive frames from each ID.
    for u = 1:uids,
        
    end
    
    epoch_idx = randsample(numsplits,numsplits);
    splits = splits(epoch_idx,:);
    same_idx = splits(splits(:,3)==1,:);
    diff_idx = splits(splits(:,3)==0,:);

    numbatch = min(size(same_idx,1),size(diff_idx,1));
    for b = 1:numbatch,
        v1 = same_idx(b,1);
        v2 = same_idx(b,2);
        v3 = diff_idx(b,1);
        v4 = diff_idx(b,2);

        if params.optjacket,
            X1b = hfunc(gsingle(X{v1}));
            X2b = hfunc(gsingle(X{v2}));
            X3b = hfunc(gsingle(X{v3}));
            X4b = hfunc(gsingle(X{v4}));
        else
            X1b = hfunc(single(X{v1}));
            X2b = hfunc(single(X{v2}));
            X3b = hfunc(single(X{v3}));
            X4b = hfunc(single(X{v4}));
        end

        [ grad, costs ] = mgrad_videos2(X1b,X2b,X3b,X4b,weights,params);
        weights.visfac = weights.visfac - lr*grad.visfac;
        weights.hidfac_a = weights.hidfac_a - lr*grad.hidfac_a;
        weights.hidfac_b = weights.hidfac_b - lr*grad.hidfac_b;
        weights.vishid_a = weights.vishid_a - lr*grad.vishid_a;
        weights.vishid_b = weights.vishid_b - lr*grad.vishid_b;
        weights.visbias = weights.visbias - lr*grad.visbias;
        weights.hidbias_a = weights.hidbias_a - lr*grad.hidbias_a;
        weights.hidbias_b = weights.hidbias_b - lr*grad.hidbias_b;

        % For visualization.
        if mod(b,10)==0,
            fprintf(1,'batch %d of %d cost=%.4f, recon=%.4f, inv=%.4f, slow=%.4f, dev=%f\n', ...
                b, numbatch, costs.total, costs.recon, costs.inv, costs.slow, costs.dev);
            if strcmp(params.l1,'rbm'),
                ns = min(16,size(X1b,2));
                dvis = X1b(:,1:ns);
                Ha = sigmoid(bsxfun(@plus,weights.vishid_a*dvis,weights.hidbias_a));
                Hb = sigmoid(bsxfun(@plus,weights.vishid_b*dvis,weights.hidbias_b));
                recon = bsxfun(@plus,weights.visfac'*((weights.hidfac_a*Ha).*(weights.hidfac_b*Hb)),weights.visbias);
                Wvis1 = weights.vishid_a(1:min(size(weights.vishid_a,1),16),:)';
                Wvis2 = weights.vishid_b(1:min(size(weights.vishid_b,1),16),:)';


                subplot(1,4,1);
                display_network_nonsquare(double(vfunc(100*Wvis1)));
                subplot(1,4,2);
                display_network_nonsquare(double(vfunc(100*Wvis2)));
                subplot(1,4,3);
                display_network_nonsquare(double(vfunc(dvis)));
                subplot(1,4,4);
                display_network_nonsquare(double(vfunc(recon)));
                print('-dpng',params.fname_png);
            end
        end
    end

    fprintf(1, 'Epoch %d of %d error = %.4f\n', i, maxiter, costs.total);
    % Checkpoint.
    tmp = weights;
    weights = gpu2cpu_struct(tmp);
    save(params.fname_mat,'weights','params');
    weights = tmp;
end

weights = gpu2cpu_struct(weights);
save(params.fname_mat,'weights','params');

end