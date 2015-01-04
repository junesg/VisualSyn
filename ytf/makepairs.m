function pairs = makepairs(ids)
    uids = unique(ids);
    pairs = [];

    for i = 1:length(uids),
        id = uids(i);
        matches = find(ids==id);
        nonmatches = find(ids~=id);
        num_matches = length(matches);

        if num_matches <= 1,
            continue;
        end
        mpairs = combnk(matches, 2);
        numcombos = size(mpairs,1);
        newpairs = [ mpairs, randsample(nonmatches,numcombos,1) ];
        pairs = [ pairs; newpairs ];
    end
end
