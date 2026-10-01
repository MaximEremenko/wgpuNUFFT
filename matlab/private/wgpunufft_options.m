function values = wgpunufft_options(opts)
% The options vector of the MEX gateway: [backend precision mode_order threads sigma].
%
% opts fields, all optional:
%   backend    'auto' (the GPU when there is one, default), 'gpu' or 'cpu'
%   precision  'auto' (default: native f64 on GPUs that have it, else Df64,
%              for double arrays; f32 for single arrays), 'f64', 'df64' or 'f32'
%   modeord    0 for centered modes (default), 1 for FFT order
%   threads    CPU threads, 0 for all (default)
%   sigma      upsampling factor, 0 for the default 2
values = [0 0 0 0 0];
if nargin < 1 || isempty(opts)
    return
end
if ~isstruct(opts)
    error('wgpunufft:input', 'opts must be a struct');
end
if isfield(opts, 'backend')
    values(1) = lookup(opts.backend, {'auto', 'gpu', 'cpu'}, 'backend');
end
if isfield(opts, 'precision')
    values(2) = lookup(opts.precision, {'auto', 'f64', 'df64', 'f32'}, 'precision');
end
if isfield(opts, 'modeord')
    values(3) = double(opts.modeord);
end
if isfield(opts, 'threads')
    values(4) = double(opts.threads);
end
if isfield(opts, 'sigma')
    values(5) = double(opts.sigma);
end
end

function index = lookup(value, names, field)
if isnumeric(value)
    index = double(value);
    return
end
index = find(strcmpi(value, names), 1) - 1;
if isempty(index)
    error('wgpunufft:input', 'opts.%s must be one of: %s', field, strjoin(names, ', '));
end
end
