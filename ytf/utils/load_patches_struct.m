function patches = load_patches_struct(xtrain,numPatches,ws,numch)

nimg = length(xtrain);
% extract random patches
k = 0;
patches = zeros(ws*ws*numch, numPatches);
for j = 1:nimg,
    fprintf('Extracting patch: %d / %d\n', j, nimg);
    nPatchperImg = ceil(numPatches/nimg);
    curimg = xtrain{j};
    curimg = curimg - mean(curimg(:));
    [rows, cols, ~] = size(curimg);
    for i = 1:nPatchperImg,
        k = k+1;
        r = randi(rows - ws + 1);
        c = randi(cols - ws + 1);
        patch = curimg(r:r+ws-1,c:c+ws-1,:);
        patches(:,k) = patch(:);
    end
end

return