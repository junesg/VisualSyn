function close_jacket(gpu_name, user_name)

if isempty(gpu_name),
    return;
end

% designate path
GPUPATH = '/mnt/neocortex/scratch/gpu_usage/';
if ~exist('user_name', 'var'),
    [~, user_name] = system('whoami');
    user_name = strtrim(user_name);
end

delete([GPUPATH gpu_name]);
delete([GPUPATH user_name '/' gpu_name]);
clear gpu_hook; % release gpu memory

fprintf('%s closed\n', gpu_name);

return;

