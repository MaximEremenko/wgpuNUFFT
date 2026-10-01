classdef wgpunufft_plan < handle
%WGPUNUFFT_PLAN  A reusable NUFFT plan of type 1, 2 or 3, on the GPU or the CPU.
%
%   plan = wgpunufft_plan(type, n_modes, isign, ntrans, eps, opts)
%   plan.setpts(x, y, z, s, t, u)
%   out = plan.execute(in)
%
%   type 1: f(k) = sum_j c(j) exp(i*isign*k.x(j))     (points to modes)
%   type 2: c(j) = sum_k f(k) exp(i*isign*k.x(j))     (modes to points)
%   type 3: f(k) = sum_j c(j) exp(i*isign*s(k).x(j))  (points to frequencies)
%
% n_modes holds the modes per axis for types 1 and 2, and the dimension for
% type 3. ntrans transforms share each point set: execute takes c as an
% M-by-ntrans array (types 1 and 3) or f as an [n_modes ntrans] array
% (type 2). Modes are in centered order, frequencies -floor(n/2) first.
%
% setpts takes one coordinate column per axis, at the positions x, y, z,
% and for type 3 the target frequencies at s, t, u; leave unused ones out
% or pass []. Plans of more than three dimensions take an M-by-dim matrix X
% (and an N-by-dim matrix S): plan.setpts(X, S). Type-1 and type-2
% coordinates are radians in [-3*pi, 3*pi].
%
% opts fields, all optional:
%   floatprec  'double' (default) or 'single': the class of the arrays
%   backend    'auto' (the GPU when there is one, default), 'gpu' or 'cpu'
%   precision  'auto' (default): native f64 on GPUs that have it, Df64
%              (about 44-48 bits) on others, f64 on the CPU, and f32 for
%              single arrays; or 'f64', 'df64', 'f32'
%   modeord    0 for centered modes (default), 1 for FFT order
%   threads    CPU threads, 0 for all (default)
%   sigma      upsampling factor, 0 for the default 2
%   adapter_name        the GPU adapter by name, such as the name CUDA
%                       reports for a device; unset, wgpu picks the adapter
%   adapter_pci_bus_id  the GPU adapter by PCI address, such as '0000:01:00.0'
%   adapter_index       the adapter_index-th adapter that matches, in the
%                       order of wgpunufft_adapters; tells identical cards apart
%
% A selected adapter that does not exist is an error, with the automatic
% backend too; CPU plans ignore the selection. The process has one GPU
% device, on the adapter of its first GPU plan: selecting another adapter is
% an error until wgpunufft_shutdown.
%
% plan.backend and plan.arithmetic report what the plan runs on.

    properties (SetAccess = private)
        type        % 1, 2 or 3
        n_modes     % modes per axis (types 1 and 2)
        dim         % dimensions
        isign       % sign of the exponent
        ntrans      % transforms per execution
        tol         % requested accuracy
        floatprec   % 'double' or 'single'
        M = 0       % points of the last setpts
        N = 0       % type-3 target frequencies of the last setpts
    end

    properties (Access = private)
        handle = []
    end

    methods
        function plan = wgpunufft_plan(type, n_modes, isign, ntrans, eps, opts)
            if nargin < 6
                opts = struct();
            end
            if nargin < 4 || isempty(ntrans)
                ntrans = 1;
            end
            if nargin < 5 || isempty(eps)
                eps = 1e-6;
            end
            plan.type = double(type);
            plan.isign = double(isign);
            plan.ntrans = double(ntrans);
            plan.tol = double(eps);
            plan.floatprec = 'double';
            if isfield(opts, 'floatprec')
                plan.floatprec = lower(opts.floatprec);
                opts = rmfield(opts, 'floatprec');
            end
            if ~any(strcmp(plan.floatprec, {'double', 'single'}))
                error('wgpunufft:input', 'opts.floatprec must be ''double'' or ''single''');
            end
            if plan.type == 3
                plan.dim = double(n_modes);
                modes = zeros(1, plan.dim);
                plan.n_modes = [];
            else
                plan.n_modes = double(n_modes(:)');
                plan.dim = numel(plan.n_modes);
                modes = plan.n_modes;
            end
            plan.handle = wgpu_nufft_mex('makeplan', plan.type, modes, plan.isign, ...
                plan.ntrans, plan.tol, strcmp(plan.floatprec, 'single'), wgpunufft_options(opts));
        end

        function setpts(plan, varargin)
            %SETPTS  Sets the points, and for type 3 the target frequencies.
            if plan.dim > 3
                X = varargin{1};
                S = [];
                if plan.type == 3
                    S = varargin{2};
                end
            else
                X = plan.columns(varargin, 1:plan.dim, 'point coordinates');
                S = [];
                if plan.type == 3
                    S = plan.columns(varargin, 4:3 + plan.dim, 'target frequencies');
                end
            end
            wgpu_nufft_mex('setpts', plan.handle, X, S);
            plan.M = size(X, 1);
            plan.N = size(S, 1);
        end

        function out = execute(plan, in)
            %EXECUTE  Runs ntrans transforms on the points last set.
            if ~isa(in, plan.floatprec)
                error('wgpunufft:input', 'the input must be a %s array', plan.floatprec);
            end
            modes = plan.n_modes;
            switch plan.type
                case 1
                    expected = plan.M * plan.ntrans;
                    outsize = [modes plan.ntrans];
                case 2
                    expected = prod(plan.n_modes) * plan.ntrans;
                    outsize = [plan.M plan.ntrans];
                otherwise
                    expected = plan.M * plan.ntrans;
                    outsize = [plan.N plan.ntrans];
            end
            if numel(in) ~= expected
                error('wgpunufft:input', 'the input must hold %d values', expected);
            end
            if plan.ntrans == 1 && plan.type == 1 && numel(modes) > 1
                outsize = modes;
            end
            out = wgpu_nufft_mex('execute', plan.handle, complex(in), outsize);
        end

        function name = backend(plan)
            %BACKEND  'gpu' or 'cpu'.
            name = wgpu_nufft_mex('planinfo', plan.handle);
        end

        function name = arithmetic(plan)
            %ARITHMETIC  'f64', 'df64' or 'f32'.
            [~, name] = wgpu_nufft_mex('planinfo', plan.handle);
        end

        function delete(plan)
            if ~isempty(plan.handle)
                try
                    wgpu_nufft_mex('destroy', plan.handle);
                catch
                    % The MEX file was cleared, which destroyed the plan.
                end
                plan.handle = [];
            end
        end
    end

    methods (Access = private)
        function X = columns(plan, args, positions, name)
            X = zeros(0, numel(positions), plan.floatprec);
            for axis = 1:numel(positions)
                position = positions(axis);
                if position > numel(args) || isempty(args{position})
                    error('wgpunufft:input', 'setpts needs %d %s', numel(positions), name);
                end
                values = args{position};
                if ~isa(values, plan.floatprec) || ~isreal(values)
                    error('wgpunufft:input', 'the %s must be real %s arrays', name, plan.floatprec);
                end
                if axis == 1
                    X = zeros(numel(values), numel(positions), plan.floatprec);
                elseif numel(values) ~= size(X, 1)
                    error('wgpunufft:input', 'the %s must all have the same length', name);
                end
                X(:, axis) = values(:);
            end
        end
    end
end
