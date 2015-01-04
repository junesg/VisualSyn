
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
%labs_ex = oneofc(labs_ex(keepidx),numex);
labs_ex = labs_ex(keepidx);
labs_id = labs_id(keepidx);
folds = folds(keepidx,:);

% Normalization.
K = 0; CC = 10; EPS = 0; % for norm of CC
[data,mn,scale,~] = ncc_soft(data', CC, K, EPS);
data = data';

disp('Done.');

%% Load model.
rng('default');
rng(2);

%model = load('results/tfd_nv2304_id400_var400_l2r0.0001_eps0.01_maxit100_20141231T170617.mat');
%model = load('results/tfd_nv2304_id400_var400_l2r0.0001_eps0.01_maxit100_20141231T170700.mat');
%model = load('results/tfd_nv2304_id800_var800_l2r0.0001_eps0.01_maxit100_20141231T170732.mat');
%model = load('results/tfd_nv2304_id700_var700_l2r0.0001_eps0.01_maxit100_20141231T170900.mat');
%model = load('results/tfd_nv2304_id400_var400_l2r0.0001_eps0.01_maxit300_20141231T171337.mat');
%model = load('results/tfd_nv2304_id500_var400_l2r0.0001_eps0.01_maxit100_20141231T212648.mat');

model = load('results/tfd_nv2304_id500_var500_l2r0.0001_eps0.001_maxit200_20150101T135730.mat');

net = model.weights;
pars = model.params;

% Get data.
fold = 1;
trainidx = folds(:,fold)==1;
validx = folds(:,fold)==2;
testidx = folds(:,fold)==3;

% Train model.
Xtrain = data(:,trainidx);
Ytrain = labs_id(trainidx);
Etrain = labs_ex(trainidx);

neutrals = find(Etrain==7);
%non_neutrals = find(Etrain~=7);
non_neutrals = find(Etrain==4);
ns = 5;
sample_neutral = randsample(neutrals, ns);
sample_nonneutral = randsample(non_neutrals, ns);
X1 = Xtrain(:, sample_neutral);
X2 = Xtrain(:, sample_nonneutral);

[ Id1, Var1, Recon1 ] = net_ff_siamese(net, X1, pars);
[ Id2, Var2, Recon2 ] = net_ff_siamese(net, X2, pars);

XT = net_transfer(net, Var2, Id1, pars);
XR = net_transfer(net, Var1, Id1, pars);

Xvis = zeros(4*48, ns*48);
for i = 1:ns,
    idx = (1+(i-1)*48):(i*48);
    Xvis(:,idx) = ...
        [ reshape(X1(:,i),[48 48]); ...
          reshape(X2(:,i),[48 48]); ...
          reshape(XR(:,i),[48 48]); ...
          reshape(XT(:,i),[48 48]) ];
end

figure(1);
imagesc(Xvis,[-.7 .7]); colormap gray; axis off;
