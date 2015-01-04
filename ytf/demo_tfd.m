function demo_tfd(numhid_id, numhid_var, numhid, numvar, numid, l2reg,...
                  maxiter,optjacket,batchsize,lambda,epsilon,enc)

addpath utils;

%% Process data
disp('Processing data...');
load /mnt/neocortex/data/toronto_face/TFD_48x48.mat;
load tfd_bad_ids.mat;
data = images;
numex = 7;        % number of expressions plus one for unlabeled.
[ num_images, dim1, dim2 ] = size(data);
data_flipped = flipdim(data, 3);
data = double(reshape(permute(data, [2 3 1]), [ dim1*dim2, num_images ]));
data_flipped = double(reshape(permute(data_flipped, [2 3 1]), [ dim1*dim2, num_images ]));
data = [ data data_flipped ];
labs_ex = [ labs_ex; labs_ex ];
labs_id = [ labs_id; labs_id ];
folds = [ folds; folds ];
labs = labs_ex;
TFD_DIM = [ 48 48 ];

keepidx = (labs_ex~=-1)&(labs_id~=-1);
for b = 1:length(badids),
    keepidx = keepidx&(labs_id~=badids(b));
end
data = data(:,keepidx);
labs_ex = oneofc(labs_ex(keepidx),numex);
labs_id = labs_id(keepidx);
folds = folds(keepidx,:);

% Normalization.
K = 0; CC = 10; EPS = 0; % for norm of CC
[data,mn,scale,~] = ncc_soft(data', CC, K, EPS);
data = data';

disp('Done.');

%% Train model.
results_file = 'results_tfd.csv';
numvis = size(data,1);
params = struct('numvis', numvis, 'numhid_id', numhid_id, 'numhid_var', numhid_var, ...
	            'l2reg', l2reg, 'epsilon', epsilon, 'dataset', 'mpie', 'maxiter', maxiter, ...
                'numid', numid, 'numvar', numvar, 'numhid', numhid,...
                'batchsize',batchsize,'lambda',lambda,'enc',enc, ...
                'numsample', 2000, 'eps_decay', 0.8, 'decay_epoch', 50)
params.fname_save = sprintf('tfd_nv%d_id%d_var%d_l2r%g_eps%g_maxit%d_%s', ...
                            numvis,numhid_id,numhid_var,l2reg,epsilon,maxiter,datestr(now,30));
params.fname_mat = sprintf('results/%s.mat', params.fname_save);
params.fname_png = sprintf('images/%s.png', params.fname_save);

   
% Get data for this fold.
fold = 1;
trainidx = folds(:,fold)==1;
validx = folds(:,fold)==2;
testidx = folds(:,fold)==3;

% Train model.
Xtrain = data(:,trainidx);
Ytrain = labs_id(trainidx);

[ weights, params ] = train_siamese(Xtrain, Ytrain, params);
save(params.fname_mat,'weights','params');

end


