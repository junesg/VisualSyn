function check_grad_adv_siamese()
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
                'l2reg', 0.0, ...
                'gradcheck', 1);
weights = init_net_siamese(params,0.1);

numdata = 10;
X1 = rand(params.numvis, numdata);
X2 = rand(params.numvis, numdata);
Y = false(numdata,1);
sameidx = rand(numdata,1) < 0.5;
Y(sameidx) = true;
theta = roll(weights);

%% Check manifold gradient.
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
    tmp = unroll(theta,pars);
    net.var_comp = tmp.var_comp;
    [ grad, cost ] = grad_adv_siamese(net, pars, X1, X2, Y);
    grad = roll(grad);
end

function theta = roll(w)
    theta = [ w.var_comp(:); ];
end

function [ weights ] = unroll(theta,pars)
    weights = struct();
    idx = 1;
    weights.var_comp = reshape(theta(idx:(idx+(pars.numvar*pars.numvar)-1)), ...
                      [pars.numvar,pars.numvar]);
    idx = idx + pars.numvar*pars.numvar;
end
