
function P = softmax(W, H)

D = W*H;
Dpre = bsxfun(@minus, D, max(D, [], 1));
D = exp(Dpre);
Dsum = sum(D);
P = bsxfun(@rdivide, D, Dsum);

end