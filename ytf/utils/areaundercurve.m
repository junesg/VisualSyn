function auc=areaundercurve(x,y)
x = reshape(x, 1, numel(x));
y = reshape(y, 1, numel(y));

[x2,inds]=sort(x);
y2=y(inds);
xdiff=diff(x2);
xdiff=[x2(1),xdiff];
auc1=sum(y2.*xdiff); % upper point area
auc2=sum([0,y2([1:end-1])].*xdiff); % lower point area
auc=mean([auc1,auc2]);

end