function [flag, chkdev] = init_jacket_auto_kihyuk(optjacket,minmem)% 1,2,3,4
addpath /usr/local/jacket/engine/;

if optjacket <= 4,
    gselect(optjacket);
    flag = 0;
    chkdev = optjacket;
else
    if ~exist('minmem','var'), minmem = 8e2; end
    addpath /mnt/neocortex/library/gputest/;
    chkdev = [];
    memdev = [];
    for devno = [1,2,3,4],
        gselect(devno);
        clear gpu_hook
        
        % memory check
        r = ginfo;
        gpumem = r.gpu_free/1024^2;
        fprintf('GPU%g memory: %g MB free\n', gselect, gpumem);
        if gpumem < minmem,
            clear gpu_hook;
            continue;
        end
        %     freememratio = r.gpu_free/r.gpu_total;
        usedmemamt = r.gpu_total - r.gpu_free;
        
        % computation error check
        addpath /mnt/neocortex/library/gputest/
        [gpu_defect sigm_errlog10_list] = gputest_sigmoid_single(devno);
        
        if gpu_defect,
            warning('GPU error is too large');
            clear gpu_hook;
            continue;
        else
            chkdev = [chkdev, devno];
            memdev = [memdev, usedmemamt];
        end
    end
    
    if isempty(chkdev),
        flag = 1;
        fprintf('All GPUs are defective, busy or GPU memory is not enough\n');
        chkdev = 0;
    else
        flag = 0;
        [~,id] = min(memdev);
        chkdev = chkdev(id);
        gselect(chkdev);
        fprintf('GPU%g is selected\n\n', gselect);
    end
end

return;
