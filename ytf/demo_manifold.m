clear all;
close all;

%% Load data from two different people.
data = load('movies_split1.mat');
uids = unique(data.ids);

%% Load model.
%model = load('results/ytf_nv2304_id600_var600_l2r0.0001_eps0.01_maxit200.mat');
%model = load('results/ytf_nv2304_id600_var600_l2r0.0001_eps0.01_maxit200_20141225T015333.mat');
%model = load('results/ytf_nv2304_id800_var800_l2r1e-05_eps0.01_maxit200_20141225T020558.mat');
%model = load('results/ytf_nv2304_id800_var800_l2r1e-05_eps0.01_maxit200_20141225T022328.mat');
%model = load('results/ytf_nv2304_id800_var800_l2r0.001_eps0.01_maxit200_20141225T024031.mat');
%model = load('results/ytf_nv2304_id300_var300_l2r0.001_eps0.01_maxit200_20141228T235623.mat');
%model = load('results/ytf_nv2304_id300_var300_l2r0.0001_eps0.01_maxit100_20141229T091939.mat');
%model = load('results/ytf_nv2304_id600_var600_l2r0.0001_eps0.01_maxit100_20141229T093048.mat');
%model = load('results/ytf_nv2304_id400_var400_l2r0.001_eps0.01_maxit100_20141229T233208.mat');
%model = load('results/ytf_nv2304_id400_var400_l2r0.01_eps0.01_maxit200_20141230T000229.mat');
%model = load('results/ytf_nv2304_id400_var400_l2r0.01_eps0.01_maxit200_20141230T001405.mat');


%model = load('results/ytf_nv2304_id400_var400_l2r0.01_eps0.01_maxit200_20141230T092012.mat');
%model = load('results/ytf_nv2304_id400_var400_l2r0.001_eps0.1_maxit100_20141230T164618.mat');
model = load('results/ytf_nv2304_id400_var400_l2r0.01_eps0.01_maxit100_20141230T175905.mat');
%model = load('results/ytf_nv2304_id400_var400_l2r0.001_eps0.1_maxit100_20141230T165251.mat');
%model = load('results/ytf_nv2304_id400_var400_l2r0.001_eps0.01_maxit100_20141230T165230.mat');

net = model.weights;
params = model.params;

%%
id1 = 1;
id2 = 2;
mov1idx = find(data.ids==(uids(id1)));
mov2idx = find(data.ids==(uids(id2)));

X1 = data.movies{mov1idx(1)};
Y1 = zeros(length(uids),size(X1,2)); Y1(uids(id1),:) = 1;
X2 = data.movies{mov2idx(1)};
Y2 = zeros(length(uids),size(X2,2)); Y2(uids(id2),:) = 2;

% Traverse manifold
[ Hid_id1, Hid_var1, Id_pred1, Var1, Hid1, ReconId1, ReconPred1 ] = net_ff(net, X1, Y1, params);
[ Hid_id2, Hid_var2, Id_pred2, Var2, Hid2, ReconId2, ReconPred2 ] = net_ff(net, X2, Y2, params);

% show reconstructions as sanity check.
figure(1);
subplot(2,2,1);
display_network_nonsquare(X1(:,1:9));
subplot(2,2,2);
display_network_nonsquare(ReconId1(:,1:9));
subplot(2,2,3);
display_network_nonsquare(X2(:,1:9));
subplot(2,2,4);
display_network_nonsquare(ReconId2(:,1:9));

% Reshape X2 into a video and transfer.
numframe = min(20,size(X2,2));
X2vid = zeros([48,48*numframe]);
X1vid = 0*X2vid;

R1 = randperm(size(X1,2));
R2 = randperm(size(X2,2));
for i = 1:numframe,
    X2vid(:,(1+(i-1)*48):i*48) = reshape(X2(:,R2(i)),[48 48]);
    
    X1recon = net_transfer(net, Var2(:,R2(i)), Y1(:,1), params);
    X1vid(:,(1+(i-1)*48):i*48) = reshape(X1recon,[48 48]);
    
    X2recon = net_transfer(net, Var2(:,R2(i)), Y2(:,1), params);
    X2vid2(:,(1+(i-1)*48):i*48) = reshape(X2recon,[48 48]);
end

figure(2);
V = [ X2vid; X1vid; X2vid2 ];
imagesc(V); colormap gray; axis off;

