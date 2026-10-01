function out = wgpunufft3d1(x, y, z, c, isign, eps, ms, mt, mu, opts)
%WGPUNUFFT3D1  Type-1 NUFFT in 3D: the modes f, an array of size [ms, mt, mu].
%
%   f = wgpunufft3d1(x, y, z, c, isign, eps, ms, mt, mu, opts)
%
%   f(k) = sum_j c(j) exp(i*isign*k.x(j))
%
%   x, y, z: point coordinates, radians in [-3*pi, 3*pi], all of length M
%   c: the M complex strengths
%   ms, mt, mu: modes per axis, frequencies -floor(n/2) to ceil(n/2)-1 in
%     centered order (opts.modeord = 1 selects FFT order)
%   isign: the sign of the exponent (>= 0 positive, < 0 negative)
%   eps: the requested relative accuracy, such as 1e-6
%   opts: optional struct; see wgpunufft_plan for its fields
%
% Double arrays run in double precision, single arrays in single precision.
% Plans are kept between calls, so repeated calls with the same sizes are fast.
if nargin < 10
    opts = [];
end
out = wgpunufft_run(1, {x, y, z}, c, isign, eps, [ms, mt, mu], opts);
end
