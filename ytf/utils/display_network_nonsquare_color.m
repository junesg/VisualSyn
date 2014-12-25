function image = display_network_nonsquare_color(centroids, H, W)
centroids = centroids';
if ~exist('H','var'), H = sqrt(size(centroids,2)/3); end
if (nargin < 3)
    W = H;
end
N=size(centroids,2)/(H*W);
assert(N == 3 || N == 1);  % color and gray images

K=size(centroids,1);
COLS=round(sqrt(K));
ROWS=ceil(K / COLS);
COUNT=COLS * ROWS;

image=ones(ROWS*(H+1), COLS*(W+1), N)*100;
for i=1:size(centroids,1)
    r= floor((i-1) / COLS);
    c= mod(i-1, COLS);
    image((r*(H+1)+1):((r+1)*(H+1))-1,(c*(W+1)+1):((c+1)*(W+1))-1,:) = reshape(centroids(i,1:W*H*N),H,W,N);
end

mn=-1.5;
mx=+1.5;
image = (image - mn) / (mx - mn);
try
    imshow(image);
    drawnow();
catch
    fprintf('no show of centroids\n');
end

return;