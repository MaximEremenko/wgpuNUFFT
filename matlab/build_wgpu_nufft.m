function build_wgpu_nufft()
%BUILD_WGPU_NUFFT  Builds the MEX gateway of the MATLAB interface.
%
% Needs Rust (cargo) and a C compiler set up for MEX (mex -setup C). Builds
% the C interface of capi/ as a static library with cargo, then compiles
% private/wgpu_nufft_mex.c against it with the interleaved complex API
% (MATLAB R2018a or later). Run it from any folder; add this folder to the
% MATLAB path to use wgpunufft*.
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
run_command(sprintf('cargo build --release --locked -p wgpu-nufft-c --manifest-path "%s"', ...
    fullfile(root, 'Cargo.toml')));
% The system libraries the static library needs, as rustc reports them.
output = run_command(sprintf(['cargo rustc --release --locked -p wgpu-nufft-c ' ...
    '--manifest-path "%s" --crate-type staticlib -- --print native-static-libs'], ...
    fullfile(root, 'Cargo.toml')));
tokens = regexp(output, 'native-static-libs:([^\r\n]*)', 'tokens', 'once');
native = {};
if ~isempty(tokens)
    native = strsplit(strtrim(tokens{1}));
end
release = fullfile(root, 'target', 'release');
if ispc
    library = fullfile(release, 'wgpu_nufft_c.lib');
    native = native(endsWith(native, '.lib'));
    native = cellfun(@(name) ['-l' erase(name, '.lib')], native, 'UniformOutput', false);
    linker = {};
else
    library = fullfile(release, 'libwgpu_nufft_c.a');
    frameworks = strjoin(native(contains(native, '-framework') | ...
        [false, strcmp(native(1:end-1), '-framework')]), ' ');
    native = native(startsWith(native, '-l'));
    linker = {};
    if ~isempty(frameworks)
        linker = {['LDFLAGS=$LDFLAGS ' frameworks]};
    end
end
mex('-R2018a', '-outdir', fullfile(here, 'private'), ...
    ['-I' fullfile(root, 'capi', 'include')], ...
    fullfile(here, 'private', 'wgpu_nufft_mex.c'), library, native{:}, linker{:});
fprintf('Built %s\n', fullfile(here, 'private', ['wgpu_nufft_mex.' mexext]));
end

function output = run_command(command)
fprintf('%s\n', command);
[status, output] = system(command);
if status ~= 0
    error('wgpunufft:build', 'command failed:\n%s', output);
end
end
