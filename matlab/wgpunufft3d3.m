function out = wgpunufft3d3(x, y, z, c, isign, eps, s, t, u, opts)
%WGPUNUFFT3D3  Type-3 NUFFT in 3D: f at the N target frequencies s, t, u.
%
%   f = wgpunufft3d3(x, y, z, c, isign, eps, s, t, u, opts)
%
%   f(k) = sum_j c(j) exp(i*isign*s(k).x(j))
%
%   x, y, z: point coordinates, any finite values, all of length M
%   c: the M complex strengths
%   s, t, u: the N target frequencies per axis
%   isign: the sign of the exponent (>= 0 positive, < 0 negative)
%   eps: the requested relative accuracy, such as 1e-6
%   opts: optional struct; see wgpunufft_plan for its fields
%
% Double arrays run in double precision, single arrays in single precision.
% Plans are kept between calls, so repeated calls with the same sizes are fast.
if nargin < 10
    opts = [];
end
out = wgpunufft_run(3, {x, y, z}, c, isign, eps, {s, t, u}, opts);
end
