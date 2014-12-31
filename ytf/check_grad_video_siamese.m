function check_grad_video_siamese()
addpath utils;
addpath utils/GradCheck;

%% Setup
params = struct('numvis', 2, ...
                'numhid', 1, ...
                'numhid_id', 1, ...
                'numhid_var', 2, ...
                'numid', 2, ...
                'numvar', 2, ...
                'lambda', 1, ...
                'enc', 'sigmoid', ...
                'l2reg', 0.01, ...
                'gradcheck', 1);
weights = init_net_siamese(params,0.1);

numdata = 10;
X1 = rand(params.numvis, numdata);
X2 = rand(params.numvis, numdata);
Y = false(numdata,1);
sameidx = rand(numdata,1)<0.5;
Y(sameidx) = true;
theta = roll(weights);

%% Check gradient.
[ cost, grad ] = wrap_grad(weights, theta, X1, X2, Y, params);
numgrad = computeNumericalGradient(@(x) wrap_grad(weights, x, X1, X2, Y, params), theta);
diff = norm(numgrad - grad)/norm(numgrad+grad);

diffvec = abs(grad - numgrad);
diffstruct = unroll(diffvec,params);
fields = fieldnames(diffstruct);
for f = 1:numel(fields),
    disp(fields{f});
    disp(diffstruct.(fields{f}));
end

disp([numgrad, grad]);
disp(diff);
assert(diff < 1e-8);

end

function [ cost, grad ] = wrap_grad(net, theta, X1, X2, Y, pars)
    weights = unroll(theta, pars);
    weights.var_comp = net.var_comp;
    [ grad, cost ] = grad_video_siamese(weights, pars, X1, X2, Y);
    grad = roll(grad);
end

function theta = roll(w)          
    theta = [ w.vis_to_hid_id(:); w.vis_to_hid_var(:); ...
              w.hid_id_to_id(:); w.hid_var_to_var(:); ...
              w.id_to_hid(:); w.var_to_hid(:); ...
              w.hid_to_vis(:); ...
              w.id_comp(:); ...
              w.bias_hid_id(:); ...
              w.bias_hid_var(:); ...
              w.bias_hid(:); w.bias_vis(:) ];
end

function [ weights ] = unroll(theta,pars)
    weights = struct();
    idx = 1;
    weights.vis_to_hid_id = reshape(theta(idx:(idx+(pars.numhid_id*pars.numvis)-1)),...
                    [pars.numhid_id,pars.numvis]);
    idx = idx + pars.numvis*pars.numhid_id;
    weights.vis_to_hid_var = reshape(theta(idx:(idx+(pars.numhid_var*pars.numvis)-1)),...
                      [pars.numhid_var,pars.numvis]);
    idx = idx + pars.numhid_var*pars.numvis;
    weights.hid_id_to_id = reshape(theta(idx:(idx+(pars.numid*pars.numhid_id)-1)),...
                      [pars.numid,pars.numhid_id]);
    idx = idx + pars.numid*pars.numhid_id;
    weights.hid_var_to_var = reshape(theta(idx:(idx+(pars.numvar*pars.numhid_var)-1)), ...
                      [pars.numid,pars.numhid_var]);
    idx = idx + pars.numid*pars.numhid_var;
    weights.id_to_hid = reshape(theta(idx:(idx+(pars.numhid*pars.numid)-1)), ...
                      [pars.numhid,pars.numid]);
    idx = idx + pars.numhid*pars.numid;
    weights.var_to_hid = reshape(theta(idx:(idx+(pars.numhid*pars.numvar)-1)), ...
                      [pars.numhid,pars.numvar]);
    idx = idx + pars.numhid*pars.numvar;
    weights.hid_to_vis = reshape(theta(idx:(idx+(pars.numhid*pars.numvis)-1)), ...
                      [pars.numvis,pars.numhid]);
    idx = idx + pars.numvis*pars.numhid;
    weights.id_comp = reshape(theta(idx:(idx+(pars.numid*pars.numid)-1)), ...
                              [pars.numid,pars.numid]);
    idx = idx + pars.numid*pars.numid;
    %weights.var_comp = reshape(theta(idx:(idx+(pars.numvar*pars.numvar)-1)), ...
    %                           [pars.numvar,pars.numvar]);
    %idx = idx + pars.numvar*pars.numvar;
    weights.bias_hid_id = reshape(theta(idx:(idx+(pars.numhid_id)-1)), ...
                                  [pars.numhid_id,1]);
    idx = idx + pars.numhid_id;
    weights.bias_hid_var = reshape(theta(idx:(idx+(pars.numhid_var)-1)), ...
                                  [pars.numhid_var,1]);
    idx = idx + pars.numhid_var;
    weights.bias_hid = reshape(theta(idx:(idx+(pars.numhid)-1)), ...
                                  [pars.numhid,1]);
    idx = idx + pars.numhid;
    weights.bias_vis = reshape(theta(idx:(idx+(pars.numvis)-1)), ...
                                  [pars.numvis,1]);
    idx = idx + pars.numvis;
end

