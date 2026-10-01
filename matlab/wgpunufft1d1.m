function out = wgpunufft1d1(x, c, isign, eps, ms, opts)
%WGPUNUFFT1D1  Type-1 NUFFT in 1D: the modes f, a column of ms values.
%
%   f = wgpunufft1d1(x, c, isign, eps, ms, opts)
%
%   f(k) = sum_j c(j) exp(i*isign*k.x(j))
%
%   x: point coordinates, radians in [-3*pi, 3*pi], of length M
%   c: the M complex strengths
%   ms: modes per axis, frequencies -floor(n/2) to ceil(n/2)-1 in
%     centered order (opts.modeord = 1 selects FFT order)
%   isign: the sign of the exponent (>= 0 positive, < 0 negative)
%   eps: the requested relative accuracy, such as 1e-6
%   opts: optional struct; see wgpunufft_plan for its fields
%
% Double arrays run in double precision, single arrays in single precision.
% Plans are kept between calls, so repeated calls with the same sizes are fast.
if nargin < 6
    opts = [];
end
out = wgpunufft_run(1, {x}, c, isign, eps, [ms], opts);
end
