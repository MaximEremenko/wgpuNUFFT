function [version, gpu] = wgpunufft_info()
%WGPUNUFFT_INFO  The library version and the GPU plans run on.
%
%   [version, gpu] = wgpunufft_info()
%
% gpu names the adapter and its backend, such as 'NAME (Vulkan)', or is
% empty without a usable GPU, when plans run on the CPU. Set the WGPU_BACKEND
% environment variable (vulkan, dx12, metal) before the first call to pick
% a backend.
[version, gpu] = wgpu_nufft_mex('info');
end
