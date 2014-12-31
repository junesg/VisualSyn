function demo_ytf_train(numhid_id, numhid_var, numhid, numvar, l2reg,...
                        maxiter,optjacket,batchsize,lambda,epsilon,enc)

data = load('movies_split1.mat');

results_file = 'ytf_results.csv';

numvis = 48*48;
uid = unique(data.ids);
numid = length(uid);
params = struct('numvis', numvis, 'numhid_id', numhid_id, 'numhid_var', numhid_var, ...
	            'l2reg', l2reg, 'epsilon', epsilon, 'dataset', 'ytf', 'maxiter', maxiter, ...
                'numid', numid, 'numvar', numvar, 'numhid', numhid,...
                'batchsize',batchsize,'lambda',lambda,'enc',enc)
params.fname_save = sprintf('ytf_nv%d_id%d_var%d_l2r%g_eps%g_maxit%d_%s', ...
                            numvis,numhid_id,numhid_var,l2reg,epsilon,maxiter,datestr(now,30));
params.fname_mat = sprintf('results/%s.mat', params.fname_save);
params.fname_png = sprintf('images/%s.png', params.fname_save);

disp(params.fname_mat);

if optjacket,
    addpath /usr/local/jacket/engine;
    addpath /mnt/neocortex/library/gputest;
    [gpu_id, gpu_name] = open_jacket;
    params.optgpu = gpu_id;
    obj = onCleanup(@()close_jacket(gpu_name));
end

[ weights, params ] = train_video_adv(data.movies,data.ids,params);
save(params.fname_mat,'weights','params');

end
