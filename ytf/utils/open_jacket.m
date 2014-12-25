function [gpu_id, gpu_name] = open_jacket(numdays, option)

addpath /usr/local/jacket/engine/;

% expire after 1 day by default
if ~exist('numdays', 'var'),
    numdays = 1;
end
if numdays > 7,
    warning('maximum number of usage is 7');
end
if ~exist('option', 'var'),
	option = 1;
end


% designate path
GPUPATH = '/mnt/neocortex/scratch/gpu_usage/';

% server name and user name
[~, server_name] = system('hostname');
[~, user_name] = system('whoami');
server_name = strtrim(server_name);
user_name = strtrim(user_name);

% check whether gpus are used
a = ginfo;

if option == 5,
    for i = 1:a.gpu_count,        
        gpu_id = i;
        gselect(gpu_id);
        c = ginfo;
        
        % simple memory check
        if (c.gpu_free/c.gpu_total) < 0.7 && (c.gpu_total-c.gpu_free)/1e6 > 700,
            clear gpu_id c;
            clear gpu_hook;
            continue;
        end
        
        fprintf('gpu %d is opened\n', gpu_id);
        break;
    end
    gpu_name = [];
else
    for i = 1:a.gpu_count,
        gpu_name = sprintf('%s_gpu_%d_lock.txt', server_name, i);
        
        % check whether gpu is locked
        if exist([GPUPATH gpu_name], 'file'),
            continue;
        end
        
        gpu_id = i;
        gselect(gpu_id);
        c = ginfo;
        
        % simple memory check
        if (c.gpu_free/c.gpu_total) < 0.7,
            clear gpu_id c;
            clear gpu_hook;
            continue;
        end
        
        
        % if there is unlocked gpu, write log file for global purpose
        fid = fopen([GPUPATH gpu_name], 'w');
        
        open_date = datestr(now, 1);
        exp_date = datestr(datenum(open_date) + numdays + 1, 1);
        
        fprintf(fid, 'user_name\t %s\n', user_name);    % user name
        fprintf(fid, 'program\t MATLAB\n');             % program
        fprintf(fid, 'start_date\t %s\n', open_date);   % start date
        fprintf(fid, 'exp_date\t %s\n', exp_date);      % expiration date
        
        fclose(fid);
        
        % write log file for each user
        if ~exist([GPUPATH user_name], 'dir'),
            mkdir([GPUPATH user_name]);
        end
        
        fid = fopen([GPUPATH user_name '/' gpu_name], 'w');
        
        fprintf(fid, 'user_name\t %s\n', user_name);    % user name
        fprintf(fid, 'program\t MATLAB\n');             % program
        fprintf(fid, 'start_date\t %s\n', open_date);   % start date
        fprintf(fid, 'exp_date\t %s\n', exp_date);      % expiration date
        
        fclose(fid);
        
        break;
    end
    
    if ~exist('gpu_id', 'var'),
        gpu_id = 0;
        gpu_name = [];
        fprintf('all gpus are locked at this moment\n');
        fprintf('please use cpu or move to other server\n');
    else
        fprintf('%s opened\n', gpu_name);
    end
end



return;

