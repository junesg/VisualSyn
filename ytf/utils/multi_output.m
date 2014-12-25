function y_mult = multi_output(y,num_out)
%%%
%   y       : 1       x batchsize
%   y_mult  : num_out x batchsize
n = length(y);
y_mult = sparse(1:n,y,1,n,num_out,n);
y_mult = full(y_mult');
return;
