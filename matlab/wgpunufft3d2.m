function out = wgpunufft3d2(x, y, z, isign, eps, f, opts)
%WGPUNUFFT3D2  Type-2 NUFFT in 3D: the values c at the M points, from the modes f.
%
%   c = wgpunufft3d2(x, y, z, isign, eps, f, opts)
%
%   c(j) = sum_k f(k) exp(i*isign*k.x(j))
%
%   x, y, z: point coordinates, radians in [-3*pi, 3*pi], all of length M
%   f: the modes, an array of size [ms, mt, mu], centered order
%   isign: the sign of the exponent (>= 0 positive, < 0 negative)
%   eps: the requested relative accuracy, such as 1e-6
%   opts: optional struct; see wgpunufft_plan for its fields
%
% Double arrays run in double precision, single arrays in single precision.
% Plans are kept between calls, so repeated calls with the same sizes are fast.
if nargin < 7
    opts = [];
end
out = wgpunufft_run(2, {x, y, z}, f, isign, eps, [size(f, 1) size(f, 2) size(f, 3)], opts);
end
