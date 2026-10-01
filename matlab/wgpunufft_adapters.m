function adapters = wgpunufft_adapters()
%WGPUNUFFT_ADAPTERS  The GPU adapters plans can select.
%
%   adapters = wgpunufft_adapters()
%
% An n-by-1 struct array, in the order opts.adapter_index counts them, with
% fields name, backend ('Vulkan', 'Metal' or 'Dx12'), device_type
% ('DiscreteGpu', 'IntegratedGpu', ...), pci_bus_id (such as '0000:01:00.0',
% or empty where the backend does not report it) and is_default, true for
% the adapter wgpu picks when the options select none. Empty without a GPU.
%
% opts.adapter_name selects an adapter by its whole name, regardless of case;
% opts.adapter_pci_bus_id by its PCI address; opts.adapter_index takes the
% adapter_index-th of those that match. A selection that matches no adapter
% is an error, with the automatic backend too. See help wgpunufft_plan.
adapters = wgpu_nufft_mex('adapters');
end
