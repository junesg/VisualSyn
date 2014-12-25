function batch_general3e(fhandle, fname_prefix, fpars, varargin)
%BATCH_GENERAL2E    a modification of batch_general written by Prof. Lee
%   INPUT: batch_general3e(fhandle, fname_prefix, fpars, varargin)
%       fhandle is a handle object, e.g. @demo_pmrbm
%       fname_prefix is a prefix for the .started / .finished files created
%       in 'batch' folder
%       fpars is a struct that contains hyperparameters, e.g.
%       struct('pbias', [0.1 0.2 0.3], 'dataset', {'mnist',
%       'mnist_bgrand'})
%       - string arguments to be tried are put in a cell array
%       - numeric arguments are put in a 1-d vector
%       - vector/matrix arguments are put in a cell array
%
%   EXAMPLE USAGE
%   
%  batch_general3e(@demo_pmrbm, 'jan04exp', ...
%     struct('mode', {'sup', 'unsup'}, 'epsilon', [2e-3 5e-3 1e-2], ...
%        'pbias', {[0.15 0.15], [0.1 0.1]}), ...
%     'skipStarted', true, 'email', 'chansool@umich.edu');    
%  where the function "demo_pmrbm" has the following signature:
%   demo_pmrbm(pars);
%
%
%   OPTIONAL ARGUMENTS "name (default value): description"  
%       skipStarted (true): if false, then restart the experiments
%       that have been started but not finished. It's useful for the cases when
%       experiments failed while running; instead of manually deleting the
%       .started file, just set this parameter to true.
%       batchdir ('./batch'): use the specified directory to write batch log (.started and .finished) files
%       useStruct (false): if true, fhandle(struct) is called; if
%       false, fhandle(unrolled parameters), (e.g. fhandle(0.1, 'mnist')) is called.
%      
%   TODO
%       - add an optinal argument pars_fixed (pars to be passed, but not part
%       of batch temporary file, e.g. jacket number)
%       - skipFinished (true by default)
%       - randomseed ('shuffle'): specify the random seed.
%      -  in-order trial (instead of randomly shuffling the order in which
%       different numbers are tried for each parameter, shuffle the order
%       in which parameters are tuned, and preserve the per-param order.
%
%   Written by Chansoo Lee (chansool@umich.edu)
%   Last Edited by Chansoo Lee 1-4-2013

%% SETUP
pars = struct(varargin{:});
if(~isfield(pars,'skipStarted')), pars.skipStarted = true; end
if(~isfield(pars, 'batchdir')), pars.batchdir = 'batch'; end
%TODO: random seed
if(~isfield(pars, 'randomseed')), rng('shuffle'); end
if(~isfield(pars, 'useStruct')), pars.useStruct = false; end

%% set up for the recursion and run it.
str_fname = func2str(fhandle);
filename = fullfile(pars.batchdir, [fname_prefix str_fname]);


if pars.useStruct
    fpars_parsed = struct();% this gets filled up
else
    assert(mod(length(fpars),2) == 0, 'when using non-struct mode, length(fpars) must be even');
    fpars_parsed = {length(fpars) / 2}; % this gets filled up
end

batch_recursion(filename, fpars, 1);

function batch_recursion(filename, fpars, i_fpar)
	names = fieldnames(fpars);
	valname = names{i_fpar};
  vallist = fpars.(valname);
  for val = vallist(randperm(length(vallist))) %vallist
    % check if the same file existsame, val);
            
    if iscell(val) % the code supports empty argument
      filename2 = sprintf('%s_%s%s', filename, valname, val{1});
      if strcmpi(val{1}, '[]')
        theVal = [];
      else
        theVal = val{1};
      end
    else
      if isfloat(val)
        filename2 = sprintf('%s_%s%0.2g', filename, valname, val);
      elseif isinteger(val)
        filename2 = sprintf('%s_%s%d', filename, valname, val);
      elseif ischar(val)
        filename2 = sprintf('%s_%s%s', filename, valname, val);
      elseif islogical(val)
        filename2 = sprintf('%s_%s%d', filename, valname, val);
      end
      theVal = val;
    end

    fpars_parsed.(valname) = theVal;
		fpars_parsed

    if i_fpar < numel(names)
      batch_recursion(filename2, fpars, i_fpar + 1);
    else                
      log_fname_started= sprintf('%s.started', filename2);
      log_fname_finished= sprintf('%s.finished', filename2);
               
			log_fname_started
      %% TODO: check the date and decide
      if exist(log_fname_finished, 'file') || (exist(log_fname_started, 'file') && pars.skipStarted)
        fprintf('file %s exists.. skip!\n', filename2);
        return;
      end
                
      %% execute
      system(sprintf('touch %s', log_fname_started));
      fprintf('Running %s\n', log_fname_started);
      %TODO: force the function handle to return a result
      %fhandle(fpars_parsed);
                
      %% send email & clean up
      system(sprintf('rm -rf %s', log_fname_started));
      system(sprintf('touch %s', log_fname_finished));
      if isfield(pars, 'email'),
        command = sprintf('/bin/echo "batch_general: %s" | /bin/mail -s %s %s', ...
                          str_fname, fname_prefix, pars.email);
				command
        system(command);
      end
    end
  end
end
end
