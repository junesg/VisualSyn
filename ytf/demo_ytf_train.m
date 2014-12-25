function demo_ytf_train(numhid_id, numhid_var, numhid, numvar, l2reg,...
                        maxiter,optjacket,epsilon)

addpath /usr/local/jacket/engine;
addpath /mnt/neocortex/library/gputest;

data = load('movies_split1.mat');

results_file = 'ytf_results.csv';

numvis = 48*48;
numid = max(data.ids);
params = struct('numvis', numvis, 'numhid_id', numhid_id, 'numhid_var', numhid_var, ...
	             'l2reg', l2reg, 'epsilon', epsilon, 'dataset', 'ytf', 'maxiter', maxiter, ...
                 'numid', numid, 'numvar', numvar, 'numhid', numhid)
params.fname_save = sprintf('ytf_nv%d_id%d_var%d_l2r%g_eps%g_maxit%d', ...
                            numvis,numhid_id,numhid_var,l2reg,epsilon,maxiter);
params.fname_mat = sprintf('results/%s.mat', params.fname_save);
params.fname_png = sprintf('images/%s.png', params.fname_save);

disp(params.fname_mat);

if optjacket,
    [gpu_id, gpu_name] = open_jacket;
    params.optgpu = gpu_id;
    obj = onCleanup(@()close_jacket(gpu_name));
end

[ weights, params ] = train_video(data.movies,data.ids,params);
save(params.fname_mat,'weights','params');

end
