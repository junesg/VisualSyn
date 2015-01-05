%% Load data.
addpath('utils');
addpath('/mnt/neocortex//library/liblinear-1.9/matlab');
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
%labs_ex = oneofc(labs_ex(keepidx),numex);
labs_ex = labs_ex(keepidx);
labs_id = labs_id(keepidx);
folds = folds(keepidx,:);

% Normalization.
K = 0; CC = 10; EPS = 0; % for norm of CC
[data,mn,scale,~] = ncc_soft(data', CC, K, EPS);
data = data';


fold = 1;
trainidx = folds(:,fold)==1;
validx = folds(:,fold)==2;
testidx = folds(:,fold)==3;

%% Train model.
Xtrain = data(:,trainidx);
Ytrain = labs_id(trainidx);
Etrain = labs_ex(trainidx);

Xval = data(:,validx);
Yval = labs_id(validx);
Eval = labs_ex(validx);
pairs_val = makepairs(Yval);

Xtest = data(:,testidx);
Ytest = labs_id(testidx);
Etest = labs_ex(testidx);
pairs_test = makepairs(Ytest);

disp('Done.');

%% Evaluation.
%model = load('results/tfd_nv2304_id500_var500_l2r0.0001_eps0.01_maxit100_20150104T193808.mat');
%model = load('results/tfd_nv2304_id10_var10_l2r0.0001_eps0.01_maxit100_20150104T213441.mat');
%model = load('results/tfd_nv2304_id10_var10_l2r0.0001_eps0.01_maxit100_20150104T213250.mat');
%model = load('results/tfd_nv2304_id20_var20_l2r0.0001_eps0.01_maxit100_20150104T231841.mat');
model = load('results/tfd_nv2304_id10_var10_l2r0.0001_eps0.01_maxit100_20150105T000224.mat');
net = model.weights;
pars = model.params;

[ IdTrain, VarTrain, ~ ] = net_ff_siamese(net, Xtrain, pars);
[ IdVal, VarVal, ~ ] = net_ff_siamese(net, Xval, pars);
[ IdTest, VarTest, ~ ] = net_ff_siamese(net, Xtest, pars);

Clist = [ 10 ];

% ID -> Var
[acc_val_id2var, acc_test_id2var] = eval_cls(Clist, IdTrain, Etrain, IdVal, Eval, IdTest, Etest);
[acc_val_var2var, acc_test_var2var] = eval_cls(Clist, VarTrain, Etrain, VarVal, Eval, VarTest, Etest);
[ auc_val_id2id ] = eval_vrf(pairs_val, IdVal);
[ auc_test_id2id ] = eval_vrf(pairs_test, IdTest);
[ auc_val_var2id ] = eval_vrf(pairs_val, VarVal);
[ auc_test_var2id ] = eval_vrf(pairs_test, VarTest);

%
fprintf(1,'\n\n\n');
fprintf(1,'VALIDATION: id2var: %.4f, var2var: %.4f, id2id: %.4f, var2id: %.4f\n', ...
        acc_val_id2var, acc_val_var2var, auc_val_id2id, auc_val_var2id);
fprintf(1,'TEST      : id2var: %.4f, var2var:%.4f, id2id: %.4f, var2id: %.4f\n', ...
        acc_test_id2var, acc_test_var2var, auc_test_id2id, auc_test_var2id);

