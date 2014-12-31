function check_grad_adv()
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
weights = init_net(params,0.1);

numdata = 10;
X = rand(params.numvis, numdata);
Y = false(params.numid, numdata);
for i = 1:numdata,
  Y(randi(params.numid,1),i) = true;
end
theta = roll(weights);

%% Check manifold gradient.
[ cost, grad ] = wrap_grad(weights, theta, X, Y, params);
numgrad = computeNumericalGradient(@(x) wrap_grad(weights, x, X, Y, params), theta);
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

function [ cost, grad ] = wrap_grad(net, theta, X, Y, pars)
    tmp = unroll(theta,pars);
    net.var_to_id = tmp.var_to_id;
    [ grad, cost ] = grad_adv(net, pars, X, Y);
    grad = roll(grad);
end

function theta = roll(w)
    theta = [ w.var_to_id(:); ];
end

function [ weights ] = unroll(theta,pars)
    weights = struct();
    idx = 1;
    weights.var_to_id = reshape(theta(idx:(idx+(pars.numid*pars.numvar)-1)), ...
                      [pars.numid,pars.numvar]);
    idx = idx + pars.numid*pars.numvar;
end
