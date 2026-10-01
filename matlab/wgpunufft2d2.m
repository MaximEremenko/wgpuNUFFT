function out = wgpunufft2d2(x, y, isign, eps, f, opts)
%WGPUNUFFT2D2  Type-2 NUFFT in 2D: the values c at the M points, from the modes f.
%
%   c = wgpunufft2d2(x, y, isign, eps, f, opts)
%
%   c(j) = sum_k f(k) exp(i*isign*k.x(j))
%
%   x, y: point coordinates, radians in [-3*pi, 3*pi], all of length M
%   f: the modes, an array of size [ms, mt], centered order
%   isign: the sign of the exponent (>= 0 positive, < 0 negative)
%   eps: the requested relative accuracy, such as 1e-6
%   opts: optional struct; see wgpunufft_plan for its fields
%
% Double arrays run in double precision, single arrays in single precision.
% Plans are kept between calls, so repeated calls with the same sizes are fast.
if nargin < 6
    opts = [];
end
out = wgpunufft_run(2, {x, y}, f, isign, eps, [size(f, 1) size(f, 2)], opts);
end
