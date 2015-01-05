function [ auc_train, auc_val, auc_test ] = eval_verify(model, pdata, h1, fold, L)
addpath /mnt/neocortex/library/liblinear-1.7/matlab;

if exist('L','var'),
    sup = true;
else
    sup = false;
end

% Do it on CPU.
model.Wv = double(model.Wv);
model.Wh = double(model.Wh);
model.Wt = double(model.Wt);
model.hbias = double(model.hbias);
model.vbias = double(model.vbias);
model.cbias = double(model.cbias);
pdata.images = double(pdata.images);
h1 = double(h1);

% Extract h2, c2;
if sup,
    c2 = sigmoid(bsxfun(@plus, model.Wc*L, model.cbias));
else
    c2 = sigmoid(repmat(model.cbias,1,size(h1,2)));
end
kmf = model.params.kmf;
for k = 1:kmf,
    h2 = sigmoid(bsxfun(@plus, model.Wh'*((model.Wv*h1).*(model.Wt*c2)), model.hbias));
    if sup,
        c2 = sigmoid(bsxfun(@plus, model.Wt'*((model.Wv*h1).*(model.Wh*h2)) + model.Wc*L, model.cbias));
    else
        c2 = sigmoid(bsxfun(@plus, model.Wt'*((model.Wv*h1).*(model.Wh*h2)), model.cbias));
    end
end

% Train
dleft = pdata.images(:,pdata.trainpairs(:,1,fold));
h1left = h1(:,pdata.trainpairs(:,1,fold));
h2left = h2(:,pdata.trainpairs(:,1,fold));
dright = pdata.images(:, pdata.trainpairs(:,2,fold));
h1right = h1(:,pdata.trainpairs(:,2,fold));
h2right = h2(:,pdata.trainpairs(:,2,fold));
trainlabels = pdata.trainlabels(:,fold);
dh1 = abs(h1left - h1right);
dh2 = abs(h2left - h2right);
mirbmtrain = [ dh1 ; dh2; ];

% Val
dleft = pdata.images(:,pdata.valpairs(:,1,fold));
h1left = h1(:,pdata.valpairs(:,1,fold));
h2left = h2(:,pdata.valpairs(:,1,fold));
dright = pdata.images(:, pdata.valpairs(:,2,fold));
h1right = h1(:,pdata.valpairs(:,2,fold));
h2right = h2(:,pdata.valpairs(:,2,fold));
vallabels = pdata.vallabels(:,fold);
dh1 = abs(h1left - h1right);
dh2 = abs(h2left - h2right);
mirbmval = [ dh1 ; dh2; ];

% Test
dleft = pdata.images(:,pdata.testpairs(:,1,fold));
h1left = h1(:,pdata.testpairs(:,1,fold));
h2left = h2(:,pdata.testpairs(:,1,fold));
dright = pdata.images(:, pdata.testpairs(:,2,fold));
h1right = h1(:,pdata.testpairs(:,2,fold));
h2right = h2(:,pdata.testpairs(:,2,fold));
testlabels = pdata.testlabels(:,fold);
dh1 = abs(h1left - h1right);
dh2 = abs(h2left - h2right);
mirbmtest = [ dh1 ; dh2; ];

cvals = [ 0.001, 0.01, 0.1 ];

auc_mirbm_train = zeros(1,length(cvals));
auc_mirbm_val = zeros(1,length(cvals));
auc_mirbm_test = zeros(1,length(cvals));
for c = 1:length(cvals),
    C = cvals(c);
    classifier = train(trainlabels, sparse(mirbmtrain), sprintf('-s 3 -c %g -q', C), 'col');
    trainscore = svm_plotroc(trainlabels,sparse(mirbmtrain),classifier);
    valscore = svm_plotroc(vallabels,sparse(mirbmval),classifier);
    testscore = svm_plotroc(testlabels,sparse(mirbmtest),classifier);
    
    auc_mirbm_train(c) = trainscore;
    auc_mirbm_val(c) = valscore;
    auc_mirbm_test(c) = testscore;
end
best_cidx = find(auc_mirbm_val == max(auc_mirbm_val));
best_cidx = best_cidx(1);
bestC = cvals(best_cidx);
auc_train = auc_mirbm_train(best_cidx);
auc_val = auc_mirbm_val(best_cidx);
auc_test = auc_mirbm_test(best_cidx);

%fprintf('auc_train=%.4g, auc_val=%.4g, auc_test=%.4g\n', auc_train, auc_val, auc_test);
