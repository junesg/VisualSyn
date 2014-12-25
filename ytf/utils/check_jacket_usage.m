function [optjacket, check_name] = check_jacket_usage(optjacket)
fname = '/home/kihyuks/jacket_usage/';
if ~exist('/home/kihyuks/jacket_usage/','dir'),
    try
        mkdir('/home/kihyuks/jacket_usage/');
    catch
        mkdir('/z/home/kihyuks/jacket_usage/');
        fname = '/z/home/kihyuks/jacket_usage/';
    end
end

a = ginfo;
if a.gpu_count == 1 && optjacket,
    optjacket = 1;
end

if ~exist('optjacket','var') || optjacket > a.gpu_count,
    for i = a.gpu_count:-1:1,
        check_name = sprintf('%s/jacket_%d.txt',fname,i);
        if ~exist(check_name,'file'),
            fid = fopen(sprintf('%s/jacket_%d.txt',fname,i),'w');
            fclose(fid);
            optjacket = i;
            break;
        end
    end
    
    if ~exist('fid','var'),
        optjacket = 0;
    end
elseif optjacket,
    check_name = sprintf('%s/jacket_%d.txt',fname,optjacket);
else
    check_name = [];
end

if optjacket <= a.gpu_count && optjacket >= 1,
    try
        gselect(optjacket);
        fprintf('%d gpu has been initialized\n', optjacket);
    catch
        optjacket = 0;
        check_name = [];
    end
end

return;
