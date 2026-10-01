function out = wgpunufft1d2(x, isign, eps, f, opts)
%WGPUNUFFT1D2  Type-2 NUFFT in 1D: the values c at the M points, from the modes f.
%
%   c = wgpunufft1d2(x, isign, eps, f, opts)
%
%   c(j) = sum_k f(k) exp(i*isign*k.x(j))
%
%   x: point coordinates, radians in [-3*pi, 3*pi], of length M
%   f: the modes, an array of size [ms], centered order
%   isign: the sign of the exponent (>= 0 positive, < 0 negative)
%   eps: the requested relative accuracy, such as 1e-6
%   opts: optional struct; see wgpunufft_plan for its fields
%
% Double arrays run in double precision, single arrays in single precision.
% Plans are kept between calls, so repeated calls with the same sizes are fast.
if nargin < 5
    opts = [];
end
out = wgpunufft_run(2, {x}, f, isign, eps, numel(f), opts);
end
