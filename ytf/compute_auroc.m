% =====================================================
% deci = score for each instance
% label_y = +1 for positive, -1 for negative (default)
% =====================================================

function auc = compute_auroc(deci,label_y,showcurve)

if ~exist('showcurve', 'var'),
    showcurve = 0;
end

if length(unique(label_y)) > 2,
    error('not binary classification');
end
if isempty(setdiff(label_y, [0 1])),
    label_y(label_y == 0) = -1; % convert to +1, -1 indices
end

if ~showcurve,
    [~, ind] = sort(deci,'descend');
    roc_y = label_y(ind);
    stack_x = cumsum(roc_y == -1)/sum(roc_y == -1);
    stack_y = cumsum(roc_y == 1)/sum(roc_y == 1);
    auc = sum((stack_x(2:length(roc_y),1)-stack_x(1:length(roc_y)-1,1)).*stack_y(2:length(roc_y),1));
else
    [stack_x, stack_y, ~, auc]=perfcurve(label_y,deci,1);
    plot(stack_x,stack_y);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC curve of (AUC = ' num2str(auc) ' )']);
end

return
