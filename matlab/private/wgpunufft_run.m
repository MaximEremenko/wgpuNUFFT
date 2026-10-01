function out = wgpunufft_run(type, coords, in, isign, eps, last, opts)
% One transform through the gateway's kept plans. coords holds the point
% coordinates per axis; last holds the mode counts (types 1 and 2) or the
% target frequencies per axis (type 3).
if nargin < 7
    opts = [];
end
class_name = class(coords{1});
if ~any(strcmp(class_name, {'double', 'single'}))
    error('wgpunufft:input', 'the coordinates must be double or single');
end
X = points(coords, class_name, 'the points');
if type == 3
    last = points(last, class_name, 'the target frequencies');
else
    last = double(last(:)');
end
if ~isa(in, class_name)
    error('wgpunufft:input', 'the %s must be %s like the points', ...
        ternary(type == 2, 'modes', 'strengths'), class_name);
end
out = wgpu_nufft_mex('simple', type, X, complex(in(:)), double(isign), double(eps), last, ...
    wgpunufft_options(opts));
end

function X = points(coords, class_name, name)
count = numel(coords{1});
X = zeros(count, numel(coords), class_name);
for axis = 1:numel(coords)
    if ~isa(coords{axis}, class_name) || ~isreal(coords{axis})
        error('wgpunufft:input', '%s must all be real %s arrays', name, class_name);
    end
    if numel(coords{axis}) ~= count
        error('wgpunufft:input', '%s must all have the same length', name);
    end
    X(:, axis) = coords{axis}(:);
end
end

function value = ternary(condition, yes, no)
if condition
    value = yes;
else
    value = no;
end
end
