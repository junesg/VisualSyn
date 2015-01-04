function check_grad_adv_var()
addpath utils;
addpath utils/GradCheck;

%% Setup
params = struct('numvis', 2, ...
                'numhid', 2, ...
                'numhid_id', 1, ...
                'numhid_var', 2, ...
                'numid', 2, ...
                'numvar', 1, ...
                'lambda', 1, ...
                'enc', 'sigmoid', ...
                'l2reg', 0.0, ...
                'gradcheck', 1);
weights = init_net_siamese(params,1);

numdata = 10;
X1 = rand(params.numvis, numdata);
X2 = rand(params.numvis, numdata);
X3 = rand(params.numvis, numdata);
Y = false(numdata,1);
sameidx = rand(numdata,1) < 0.5;
Y(sameidx) = true;
theta = roll(weights);

%% Check manifold gradient.
[ cost, grad ] = wrap_grad(weights, theta, X1, X2, X3, Y, params);
numgrad = computeNumericalGradient(@(x) wrap_grad(weights, x, X1, X2, X3, Y, params), theta);
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

function [ cost, grad ] = wrap_grad(net, theta, X1, X2, X3, Y, pars)
    tmp = unroll(theta,pars);
    net.id_comp = tmp.id_comp;
    [ grad, cost ] = grad_adv_var(net, pars, X1, X2, X3, Y);
    grad = roll(grad);
end

function theta = roll(w)
    theta = [ w.id_comp(:); ];
end

function [ weights ] = unroll(theta,pars)
    weights = struct();
    idx = 1;
    weights.id_comp = reshape(theta(idx:(idx+(pars.numid*pars.numid)-1)), ...
                      [pars.numid,pars.numid]);
    idx = idx + pars.numid*pars.numid;
end
