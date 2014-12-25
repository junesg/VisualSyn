function [patches M P] = load_patches(xtrain,numPatches,ws,numch,optZCA)

rs = sqrt(size(xtrain,1));
% extract random patches
patches = zeros(numPatches, ws*ws*numch);
for i=1:numPatches
    if (mod(i,10000) == 0), fprintf('Extracting patch: %d / %d\n', i, numPatches); end
    r = random('unid', rs - ws + 1);
    c = random('unid', rs - ws + 1);
    patch = reshape(xtrain(:,mod(i-1,size(xtrain,2))+1), rs,rs);
    patch = patch(r:r+ws-1,c:c+ws-1);
    patches(i,:) = patch(:)';
end
% normalize for contrast
if optZCA,
    patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
    C = cov(patches);
    M = mean(patches);
    [V,D] = eig(C);
    P = V * diag(sqrt(1./(diag(D)+1e-3))) * V';
    patches = bsxfun(@minus, patches, M) * P;
else
    M = [];
    P = [];    
end

return