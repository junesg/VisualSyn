% x : numfeat x numdata
% y : 1 x numdata (or numdata x 1)
% folds : numdata x numfolds

function auc = eval_vrf(pairs, h)

% construct feature
m_ref = h(:, pairs(:, 1));
m_same = h(:, pairs(:, 2));
m_diff = h(:, pairs(:, 3));

score_pos = bsxfun(@rdivide, abs(sum(m_ref.*m_same, 1)), sqrt(bsxfun(@times, sum(m_ref.^2, 1), sum(m_same.^2, 1))+1e-6));
score_neg = bsxfun(@rdivide, abs(sum(m_ref.*m_diff, 1)), sqrt(bsxfun(@times, sum(m_ref.^2, 1), sum(m_diff.^2, 1))+1e-6));
auc_a = compute_auroc([score_pos score_neg],[ones(length(score_pos), 1) ; -ones(length(score_neg), 1)]);
auc_b = compute_auroc([score_neg score_pos],[-ones(length(score_neg), 1) ; ones(length(score_pos), 1)]);
auc = min(auc_a, auc_b);

return;
