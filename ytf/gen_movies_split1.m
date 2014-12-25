%%
meta = load('/mnt/neocortex/scratch/reedscot/data/ytfaces/meta_data/meta_and_splits.mat');
data = load('ytmovies.mat');
u1 = unique(meta.Splits(:,1:2,1));

%%
mov_idx = [];
for i = 1:length(data.movies),
    mov = data.movies{i};
    if ~isfield(mov, 'videos'), continue; end;
    for j = 1:length(mov.videos),
        mov_idx = [ mov_idx; i, j ];
    end
end


movies = cell(length(u1),1);
ids = [];
for i = 1:length(u1),
    fprintf(1,'subject %d of %d...\n', u1(i), length(u1));
    idx = u1(i);
    subject = data.movies{mov_idx(idx,1)};
    movies{i} = subject.videos{mov_idx(idx,2)};
    ids = [ ids, mov_idx(idx,1) ];
end

%%
uids = unique(ids);
for u = 1:length(uids),
    ids(ids==uids(u)) = u;
end
%%
save('movies_split1.mat', 'movies', 'ids');


