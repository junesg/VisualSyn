% Evaluate classification accuracy of the disBM features.
%
% x : numfeat x numdata
% y : 1 x numdata (or numdata x 1)
% folds : numdata x numfolds

function [acc_val, acc_test] = eval_cls(Clist, xtrain, ytrain, xval, yval, xtest, ytest)

if ~exist('Clist', 'var') || isempty(Clist),
    Clist = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 10, 30, 100, 300, 1000];
end

if length(Clist) == 1,
    % In this case, assume we want the final test accuracy (after cross-validation).
    [~, ~, acc_test] = liblinear_wrapper(Clist, [xtrain xval], [ytrain(:) ;  yval(:)], xval, yval(:), xtest, ytest(:));
    acc_val = [];
else
    % In this case, report validation accuracy for each C value.
    [~, ~, ~, ~, ~, acc_val, acc_test] = liblinear_wrapper(Clist, xtrain, ytrain(:), xval, yval(:), xtest, ytest(:));
end


return;
