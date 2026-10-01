function wgpunufft_shutdown()
%WGPUNUFFT_SHUTDOWN  Releases the GPU device and the plans one-call functions keep.
%
% Plans made with wgpunufft_plan keep the device until they are deleted.
% Clearing the MEX file (clear mex) releases everything.
wgpu_nufft_mex('shutdown');
end
