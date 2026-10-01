function test_wgpunufft(backend)
%TEST_WGPUNUFFT  Checks the MATLAB interface against direct sums.
%
%   test_wgpunufft          % the GPU when there is one, else the CPU
%   test_wgpunufft('cpu')   % or 'gpu'
%
% Build the MEX file first with build_wgpu_nufft. Errors on any failure.
if nargin < 1
    backend = 'auto';
end
opts.backend = backend;
[version, gpu] = wgpunufft_info();
fprintf('wgpu-nufft %s, GPU: %s\n', version, gpu);
tol = 1e-9;
if strcmp(backend, 'cpu') || isempty(gpu)
    tol = 1e-12;
end
check = @(name, actual, expected, limit) report(name, actual, expected, limit);
rng(7);
M = 70;
x = pi * (2 * rand(M, 1) - 1);
y = pi * (2 * rand(M, 1) - 1);
z = pi * (2 * rand(M, 1) - 1);
c = complex(randn(M, 1), randn(M, 1));

% Type 1 in 1D and 2D, type 2 in 3D, type 3 in 2D.
ms = 13; mt = 8; mu = 5;
f = wgpunufft1d1(x, c, -1, tol, ms, opts);
check('1d1', f, direct1(c, {x}, -1, ms), 10 * tol);

f = wgpunufft2d1(x, y, c, 1, tol, ms, mt, opts);
check('2d1', f, direct1(c, {x, y}, 1, [ms mt]), 10 * tol);

modes = complex(randn(ms, mt, mu), randn(ms, mt, mu));
values = wgpunufft3d2(x, y, z, 1, tol, modes, opts);
check('3d2', values, direct2(modes, {x, y, z}, 1), 10 * tol);

N = 40;
s = 9 * (2 * rand(N, 1) - 1);
t = 6 * (2 * rand(N, 1) - 1);
f = wgpunufft2d3(x, y, c, 1, tol, s, t, opts);
check('2d3', f, direct3(c, {x, y}, 1, {s, t}), 10 * tol);

% A reused plan: two point sets, three transforms at a time, FFT order.
plan_opts = opts;
plan_opts.modeord = 1;
plan = wgpunufft_plan(1, [ms mt], -1, 3, tol, plan_opts);
fprintf('plan runs on %s in %s\n', plan.backend(), plan.arithmetic());
for round = 1:2
    xr = 3 * pi * (2 * rand(M, 1) - 1);
    yr = pi * (2 * rand(M, 1) - 1);
    plan.setpts(xr, yr);
    strengths = complex(randn(M, 3), randn(M, 3));
    out = plan.execute(strengths);
    for k = 1:3
        expected = direct1(strengths(:, k), {xr, yr}, -1, [ms mt]);
        % FFT order: frequency 0 first along both axes.
        expected = circshift(expected, [-floor(ms / 2), -floor(mt / 2)]);
        check(sprintf('plan %d.%d', round, k), out(:, :, k), expected, 10 * tol);
    end
end
delete(plan);

% Single precision.
fs = wgpunufft1d1(single(x), single(c), 1, 1e-5, ms, opts);
assert(isa(fs, 'single'));
check('single 1d1', double(fs), direct1(double(single(c)), {double(single(x))}, 1, ms), 1e-4);

% Errors come back as MATLAB errors.
try
    wgpunufft1d1(10 * ones(1, 1), complex(1), 1, 1e-6, 8, opts);
    error('wgpunufft:test', 'a point outside [-3*pi, 3*pi] was accepted');
catch failure
    assert(contains(failure.message, '3*pi'), failure.message);
    fprintf('errors: ok, %s\n', failure.message);
end
wgpunufft_shutdown();
fprintf('all checks passed\n');
end

function report(name, actual, expected, limit)
error_value = norm(actual(:) - expected(:)) / norm(expected(:));
if ~(error_value <= limit)
    error('wgpunufft:test', '%s: relative error %.3g above %.3g', name, error_value, limit);
end
fprintf('%s: ok, relative error %.3g\n', name, error_value);
end

% Centered frequencies -floor(n/2) .. ceil(n/2)-1 of each axis.
function k = freqs(n)
k = (0:n - 1)' - floor(n / 2);
end

function f = direct1(c, coords, isign, n_modes)
% Column-major mode index to per-axis centered frequencies.
count = prod(n_modes);
subs = cell(1, numel(n_modes));
[subs{:}] = ind2sub([n_modes 1], (1:count)');
f = zeros(count, 1);
for j = 1:numel(c)
    phase = zeros(count, 1);
    for axis = 1:numel(n_modes)
        phase = phase + (subs{axis} - 1 - floor(n_modes(axis) / 2)) * coords{axis}(j);
    end
    f = f + c(j) * exp(1i * isign * phase);
end
f = reshape(f, [n_modes 1]);
end

function c = direct2(f, coords, isign)
n_modes = size(f);
grids = cell(1, 3);
[grids{:}] = ndgrid(freqs(n_modes(1)), freqs(n_modes(2)), freqs(n_modes(3)));
c = zeros(numel(coords{1}), 1);
for j = 1:numel(coords{1})
    phase = grids{1} * coords{1}(j) + grids{2} * coords{2}(j) + grids{3} * coords{3}(j);
    c(j) = sum(f(:) .* exp(1i * isign * phase(:)));
end
end

function f = direct3(c, coords, isign, targets)
f = zeros(numel(targets{1}), 1);
for k = 1:numel(targets{1})
    phase = zeros(numel(c), 1);
    for axis = 1:numel(coords)
        phase = phase + targets{axis}(k) * coords{axis};
    end
    f(k) = sum(c .* exp(1i * isign * phase));
end
end
