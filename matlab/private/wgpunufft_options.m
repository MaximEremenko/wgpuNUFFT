function values = wgpunufft_options(opts)
% The options struct of the MEX gateway, with the fields of wgpu_nufft_opts.
%
% opts fields, all optional:
%   backend    'auto' (the GPU when there is one, default), 'gpu' or 'cpu'
%   precision  'auto' (default: native f64 on GPUs that have it, else Df64,
%              for double arrays; f32 for single arrays), 'f64', 'df64' or 'f32'
%   modeord    0 for centered modes (default), 1 for FFT order
%   threads    CPU threads, 0 for all (default)
%   sigma      upsampling factor, 0 for the default 2
%   adapter_name, adapter_pci_bus_id, adapter_index
%              the GPU adapter: by name, by PCI address such as
%              '0000:01:00.0', and by index among the adapters that match, as
%              wgpunufft_adapters lists them; unset, wgpu picks the adapter
values = struct('backend', 0, 'precision', 0, 'mode_order', 0, 'threads', 0, ...
    'sigma', 0, 'adapter_index', 0, 'adapter_name', '', 'adapter_pci_bus_id', '');
if nargin < 1 || isempty(opts)
    return
end
if ~isstruct(opts)
    error('wgpunufft:input', 'opts must be a struct');
end
if isfield(opts, 'backend')
    values.backend = lookup(opts.backend, {'auto', 'gpu', 'cpu'}, 'backend');
end
if isfield(opts, 'precision')
    values.precision = lookup(opts.precision, {'auto', 'f64', 'df64', 'f32'}, 'precision');
end
if isfield(opts, 'modeord')
    values.mode_order = double(opts.modeord);
end
if isfield(opts, 'threads')
    values.threads = double(opts.threads);
end
if isfield(opts, 'sigma')
    values.sigma = double(opts.sigma);
end
if isfield(opts, 'adapter_index')
    values.adapter_index = double(opts.adapter_index);
end
if isfield(opts, 'adapter_name')
    values.adapter_name = characters(opts.adapter_name, 'adapter_name');
end
if isfield(opts, 'adapter_pci_bus_id')
    values.adapter_pci_bus_id = characters(opts.adapter_pci_bus_id, 'adapter_pci_bus_id');
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

function value = characters(value, field)
if isstring(value) && isscalar(value)
    value = char(value);
end
if ~(ischar(value) && (isempty(value) || isrow(value)))
    error('wgpunufft:input', 'opts.%s must be a character vector', field);
end
end
