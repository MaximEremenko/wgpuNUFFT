function out = wgpunufft1d3(x, c, isign, eps, s, opts)
%WGPUNUFFT1D3  Type-3 NUFFT in 1D: f at the N target frequencies s.
%
%   f = wgpunufft1d3(x, c, isign, eps, s, opts)
%
%   f(k) = sum_j c(j) exp(i*isign*s(k).x(j))
%
%   x: point coordinates, any finite values, of length M
%   c: the M complex strengths
%   s: the N target frequencies per axis
%   isign: the sign of the exponent (>= 0 positive, < 0 negative)
%   eps: the requested relative accuracy, such as 1e-6
%   opts: optional struct; see wgpunufft_plan for its fields
%
% Double arrays run in double precision, single arrays in single precision.
% Plans are kept between calls, so repeated calls with the same sizes are fast.
if nargin < 6
    opts = [];
end
out = wgpunufft_run(3, {x}, c, isign, eps, {s}, opts);
end
