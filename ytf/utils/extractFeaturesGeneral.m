% patches : (patch dimension) X (# of patches for given image)
% features : (feature dimension) X (# of patches for given image)
function [features] = extractFeaturesGeneral(model, patches, mode, varargin)
%             if self.standardize
%                 patches = standardize_col(patches, self.mu, self.stdev);
%             end

switch lower(mode),
    case 'hard',
        [val,label] = max(bsxfun(@minus,model.centroids'*patches,0.5*sum(model.centroids.^2,1)')); % assign samples to the nearest centers
        n = size(patches,2);
        E = sparse(1:n,label,1,n,model.k,n)';  % transform label into indicator matrix
        features = E;
        
    case 'rbf',
        % compute 'rbf' activation function
        xx = sum(patches.^2, 1); % 1*N vector (N:#examples)
        cc = sum(model.centroids.^2, 1)'; % K*1 vector
        z = sqrt( bsxfun(@plus, cc, bsxfun(@minus, xx, 2*model.centroids'*patches)) ); % distances
        
        tau = size(patches,1)*model.gmmparams.sigma^2;
        features = exp(-z.^2./tau);
        %features = max(bsxfun(@minus, mu, z), 0);
        
    case 'gmm',
        % compute 'gmm' activation function
        % Note that this assumes a shared sigma (so it's a simplicification of a full GMM)
        
        xx = sum(patches.^2, 1); % 1*N vector (N:#examples)
        cc = sum(model.centroids.^2, 1)'; % K*1 vector
        distsq = sqrt( bsxfun(@plus, cc, bsxfun(@minus, xx, 2*model.centroids'*patches)) ); % distances
        
        distsq = bsxfun(@plus, -distsq./(2*model.gmmparams.sigma^2), log(full(model.gmmparams.p0))); % - ||x-c||^2/(2sigma^2)
        % distsq = double(-distsq./(2*self.gmmparams.sigma^2)) + double(repmat(log(self.gmmparams.p0), 1, size(distsq,2)));
        distsq = bsxfun(@minus, distsq, max(distsq)); % to make it numerically stable
        distsq = exp(distsq);
        
        features = bsxfun(@rdivide, distsq, sum(distsq)); % normalize
    case 'triangle',
        % compute 'triangle' activation function
        xx = sum(patches.^2, 1); % 1*N vector (N:#examples)
        cc = sum(model.centroids.^2, 1)'; % K*1 vector
        % xc = patches * centroids';
        
        % z = sqrt( bsxfun(@plus, cc, bsxfun(@minus, xx, 2*centroids'*patches)) ); % distances
        z = sqrt( bsxfun(@plus, cc, bsxfun(@minus, xx, 2*model.centroids'*patches)) ); % distances
        
        [v,inds] = min(z,[],1);
        mu = mean(z, 1); % average distance to centroids for each patch
        features = max(bsxfun(@minus, mu, z), 0);
        
    case 'triangle_quantile',
        % compute 'triangle' activation function
        xx = sum(patches.^2, 1); % 1*N vector (N:#examples)
        cc = sum(model.centroids.^2, 1)'; % K*1 vector
        % xc = patches * centroids';
        
        % z = sqrt( bsxfun(@plus, cc, bsxfun(@minus, xx, 2*centroids'*patches)) ); % distances
        z = sqrt( bsxfun(@plus, cc, bsxfun(@minus, xx, 2*model.centroids'*patches)) ); % distances
        
        [v,inds] = min(z,[],1);
        % mu = mean(z, 1); % average distance to centroids for each patch
        mu = quantile(z, varargin{1}, 1);
        features = max(bsxfun(@minus, mu, z), 0);
        
        1;
    case 'llc',
        % just use the LLC coding paper
        features = LLC_coding_appr(model.centroids', patches', varargin{1})';
        
    case 'randomwalk',
        % patches = patches(:, 1:50);
        
        [val,label] = max(bsxfun(@minus,model.centroids'*patches,0.5*sum(model.centroids.^2,1)')); % assign samples to the nearest centers
        n = size(patches,2);
        E = sparse(1:n,label,1,n,model.k,n)';  % transform label into indicator matrix
        
        %%
        % construct P matrix
        cc = sum(model.centroids.^2, 1)'; % K*1 vector
        D = sqrt( bsxfun(@plus, cc, bsxfun(@minus, cc', 2*model.centroids'*model.centroids)) ); % pairwise distances
        sigmasq = quantile(D(:), 0.1);
        D(D>sigmasq) = inf;
        expD = exp(-0.5*D./sigmasq);
        % expD = exp(-0.5*sqrt(D)./sqrt(sigmasq));
        
        Q = expD./repmat(sum(expD,2), 1, size(expD,2)); % transition: pnew = p'*Q;
        % figure(4), imagesc(Q), colorbar
        
        Eiter= E;
        for t=1:1 %5
            Eiter = (Eiter'*Q)';
            % Eiter = max(0, (Eiter'*Q)'+ 0.05*bsxfun(@minus, E, mean(E)));
            % Eiter = Eiter./repmat(sum(Eiter,1), size(Eiter,1), 1);
        end
        features = Eiter;
        figure(100), imagesc(Eiter);
        
        %                     % feat = ((eye(size(Q))- 0.99*Q)')\(Q'*E);
        %                     feat = ((eye(size(Q))- 0.995*Q)')\(E);
        %                     feat = ((eye(size(Q))- 0.9999*Q)')\bsxfun(@minus, E, mean(E));
        %                     figure(100), imagesc(feat)
        %%
        %                     % compute 'triangle' activation function
        %                     xx = sum(patches.^2, 1); % 1*N vector (N:#examples)
        %                     cc = sum(self.centroids.^2, 1)'; % K*1 vector
        %                     % xc = patches * centroids';
        %
        %                     % z = sqrt( bsxfun(@plus, cc, bsxfun(@minus, xx, 2*centroids'*patches)) ); % distances
        %                     z = sqrt( bsxfun(@plus, cc, bsxfun(@minus, xx, 2*self.centroids'*patches)) ); % distances
        %
        %                     [v,inds] = min(z,[],1);
        %                     mu = mean(z, 1); % average distance to centroids for each patch
        %                     features = max(bsxfun(@minus, mu, z), 0);
    case 'abs_rect',
        newcenter = bsxfun(@rdivide, model.centroids, sqrt(sum((model.centroids).^2,1)));
        features = abs(newcenter'*patches); % assign samples to the nearest centers
        
end

end