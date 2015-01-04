function demo_mpie(numhid_id, numhid_var, numhid, numvar, numid, l2reg,...
                   maxiter,optjacket,batchsize,lambda,epsilon,enc)

addpath utils;
midir = '/mnt/neocortex/scratch/reedscot/code/libdeepnets2/experimental/manifold_interaction';
addpath([midir '/zmpie/multipie/prep']);
addpath([midir '/../mpie_unsup']);
addpath([midir '/../mpie_unsup/results']);

disp('Loading data...');
data = multipieLoadCropped('/mnt/neocortex/scratch/zhangyuting/manifold-interaction/MultiPie/cropped/nf_neu_48x48_resized_from_80x60.mat');
data.mpfaces = im2double(data.mpfaces(:,:,2,:) ); % use GREEN channel only
data.mpfaces = reshape(data.mpfaces, [size(data.mpfaces,1)*size(data.mpfaces,2),size(data.mpfaces,4)] );
keepidx = data.mpilabel==7;
trainidx = data.mpslabel(keepidx)==1;
validx = data.mpslabel(keepidx)==2;
testidx = data.mpslabel(keepidx)==3;

K = 0; CC = 10; EPS = 0; % for norm of CC
[mpfaces] = ncc_soft(data.mpfaces(:,keepidx)', CC, K, EPS);
mpfaces = mpfaces';
mpvlabel = data.mpvlabel(keepidx);
mpplabel = data.mpplabel(keepidx);

numview = 15;
umap = [ 110, 120, 90, 80, 81, 130, 140, 51, 50, 41, 191, 190, 200, 10, 240 ];
foo = 0*mpvlabel;
for u = 1:length(umap),    
    foo(mpvlabel==umap(u)) = u;
end
mpvlabel = foo;

fprintf(1,'Done. %d total faces\n', size(mpfaces,2));
save('mpie_processed.mat','mpfaces','mpvlabel','mpplabel','trainidx','validx','testidx');

%{
load('mpie_processed.mat');
%}

% Training.
results_file = 'mpie_results.csv';
params = struct('numvis', numvis, 'numhid_id', numhid_id, 'numhid_var', numhid_var, ...
	              'l2reg', l2reg, 'epsilon', epsilon, 'dataset', 'mpie', 'maxiter', maxiter, ...
                'numid', numid, 'numvar', numvar, 'numhid', numhid,...
                'batchsize',batchsize,'lambda',lambda,'enc',enc, ...
                'numsample', 2000)
params.fname_save = sprintf('mpie_nv%d_id%d_var%d_l2r%g_eps%g_maxit%d_%s', ...
                            numvis,numhid_id,numhid_var,l2reg,epsilon,maxiter,datestr(now,30));
params.fname_mat = sprintf('results/%s.mat', params.fname_save);
params.fname_png = sprintf('images/%s.png', params.fname_save);

Xtrain = mpfaces(:,trainidx);
Ytrain = mpplabel(trainidx);
[ weights, params ] = train_siamese(Xtrain, Ytrain, params);
save(params.fname_mat,'weights','params');

end

