function check_jacket(option)

if ~exist('option', 'var'),
    option = 'current';
end

addpath /usr/local/jacket/engine/;

% designate path
GPUPATH = '/mnt/neocortex/scratch/gpu_usage/';

if strcmp(option, 'current'),
    % server name and user name
    [~, server_name] = system('hostname');
    server_name = strtrim(server_name);
    
    % check whether gpus are used
    a = ginfo;
    
    for i = 1:a.gpu_count,
        gpu_name = sprintf('%s_gpu_%d_lock.txt', server_name, i);
        
        % check whether gpu is locked
        if exist([GPUPATH gpu_name], 'file'),
            fprintf('%s not available\n', gpu_name);
            continue;
        end
        
        gpu_id = i;
        gselect(gpu_id);
        c = ginfo;
        
        % simple memory check
        if (c.gpu_free/c.gpu_total) < 0.7 && (c.gpu_total-c.gpu_free)/1e6 > 700,
            clear gpu_id c;
            clear gpu_hook;
            fprintf('%s not available\n', gpu_name);
            continue;
        end
        
        fprintf('%s available\n', gpu_name);
    end
elseif strcmp(option, 'all'),
    gpu_lock_list = dir(GPUPATH);
    
    for i = 1:length(gpu_lock_list),
        gpu_name = gpu_lock_list(i).name;
        if strcmp(gpu_name, '.') || strcmp(gpu_name, '..') || isdir([GPUPATH gpu_name]),
            continue;
        end
        fprintf('%s is locked\n',  strtok(gpu_name, '.'));
    end
end


return;

