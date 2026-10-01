//! The GPU adapters plans can run on, and the one the options select.

use std::fmt;

use wgpu_nufft::wgpu;

use crate::error::{invalid, Error, Result};

/// The instance plans enumerate their adapters on: Vulkan, Metal and DX12,
/// unless `WGPU_BACKEND` names others.
pub(crate) fn instance() -> wgpu::Instance {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12;
    wgpu::Instance::new(descriptor.with_env())
}

/// The adapters of `instance` in wgpu's order: by backend (Vulkan, Metal,
/// DX12), then as the drivers list them.
pub(crate) fn adapters(instance: &wgpu::Instance) -> Vec<wgpu::Adapter> {
    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
}

pub(crate) fn infos(instance: &wgpu::Instance) -> Vec<wgpu::AdapterInfo> {
    adapters(instance)
        .iter()
        .map(wgpu::Adapter::get_info)
        .collect()
}

/// The adapters, each with whether it is the one wgpu picks without a
/// selection.
pub(crate) fn listed() -> Vec<(wgpu::AdapterInfo, bool)> {
    let infos = infos(&instance());
    let default = default_position(&infos);
    infos
        .into_iter()
        .enumerate()
        .map(|(position, info)| (info, Some(position) == default))
        .collect()
}

/// The adapter `request_adapter` picks for high performance: the first
/// discrete GPU, else the first integrated one, and so on.
fn default_position(infos: &[wgpu::AdapterInfo]) -> Option<usize> {
    infos
        .iter()
        .enumerate()
        .min_by_key(|&(position, info)| (preference(info.device_type), position))
        .map(|(position, _)| position)
}

fn preference(device_type: wgpu::DeviceType) -> u8 {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => 1,
        wgpu::DeviceType::IntegratedGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::VirtualGpu => 4,
        wgpu::DeviceType::Cpu => 5,
    }
}

/// How messages and `wgpu_nufft_gpu_name` name an adapter.
pub(crate) fn label(info: &wgpu::AdapterInfo) -> String {
    format!("{} ({:?})", info.name, info.backend)
}

/// A PCI address, `[domain:]bus:device[.function]` in hexadecimal as CUDA,
/// nvidia-smi and wgpu write it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PciAddress {
    domain: u32,
    bus: u32,
    device: u32,
    function: Option<u32>,
}

impl PciAddress {
    pub(crate) fn parse(text: &str) -> Option<Self> {
        let hex = |part: &str| {
            (!part.is_empty() && part.len() <= 8 && part.bytes().all(|b| b.is_ascii_hexdigit()))
                .then(|| u32::from_str_radix(part, 16).ok())
                .flatten()
        };
        let (address, function) = match text.trim().split_once('.') {
            Some((address, function)) => (address, Some(hex(function)?)),
            None => (text.trim(), None),
        };
        let parts: Vec<&str> = address.split(':').collect();
        let (domain, bus, device) = match parts[..] {
            [domain, bus, device] => (hex(domain)?, hex(bus)?, hex(device)?),
            [bus, device] => (0, hex(bus)?, hex(device)?),
            _ => return None,
        };
        Some(Self {
            domain,
            bus,
            device,
            function,
        })
    }

    /// Whether the two name one device; an address without a function
    /// matches every function.
    fn matches(&self, other: &Self) -> bool {
        (self.domain, self.bus, self.device) == (other.domain, other.bus, other.device)
            && (self.function.is_none()
                || other.function.is_none()
                || self.function == other.function)
    }
}

impl fmt::Display for PciAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}",
            self.domain, self.bus, self.device
        )?;
        match self.function {
            Some(function) => write!(f, ".{function:x}"),
            None => Ok(()),
        }
    }
}

/// The adapter the options select; all unset leaves the choice to wgpu.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct AdapterSelection {
    /// The whole name, without surrounding spaces; empty matches any.
    pub(crate) name: String,
    pub(crate) pci_bus_id: Option<PciAddress>,
    /// The position among the matching adapters, from 1; 0 for none.
    pub(crate) index: u32,
}

impl AdapterSelection {
    pub(crate) fn is_default(&self) -> bool {
        *self == Self::default()
    }

    fn matches(&self, info: &wgpu::AdapterInfo) -> bool {
        let name = self.name.is_empty()
            || info.name.trim().to_lowercase() == self.name.trim().to_lowercase();
        let address = self.pci_bus_id.is_none_or(|wanted| {
            PciAddress::parse(&info.device_pci_bus_id).is_some_and(|found| wanted.matches(&found))
        });
        name && address
    }
}

impl fmt::Display for AdapterSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if !self.name.is_empty() {
            parts.push(format!("name \"{}\"", self.name));
        }
        if let Some(address) = self.pci_bus_id {
            parts.push(format!("PCI bus ID {address}"));
        }
        if self.index > 0 {
            parts.push(format!("index {}", self.index));
        }
        if parts.is_empty() {
            f.write_str("no adapter")
        } else {
            f.write_str(&parts.join(", "))
        }
    }
}

/// The position in `infos` of the adapter `selection` names. The index
/// counts the matches in list order; without it, the matches of the first
/// backend that has any must be one adapter, as a GPU is listed once per
/// backend (Vulkan and DX12 on Windows).
pub(crate) fn resolve(infos: &[wgpu::AdapterInfo], selection: &AdapterSelection) -> Result<usize> {
    let matches: Vec<usize> = (0..infos.len())
        .filter(|&position| selection.matches(&infos[position]))
        .collect();
    let not_found = || {
        Error::GpuUnavailable(format!(
            "no GPU adapter matches {selection}; the adapters are {}",
            describe(infos, 0..infos.len())
        ))
    };
    if selection.index > 0 {
        return matches
            .get(selection.index as usize - 1)
            .copied()
            .ok_or_else(not_found);
    }
    let first = *matches.first().ok_or_else(not_found)?;
    let backend = infos[first].backend;
    let alike: Vec<usize> = matches
        .into_iter()
        .filter(|&position| infos[position].backend == backend)
        .collect();
    if alike.len() > 1 {
        return Err(invalid(format!(
            "{} {backend:?} adapters match {selection}: {}; set adapter_index or \
             adapter_pci_bus_id to choose one",
            alike.len(),
            describe(infos, alike.iter().copied())
        )));
    }
    Ok(first)
}

/// `1. NAME (Vulkan, DiscreteGpu, 0000:01:00.0); 2. ...`, numbered from 1
/// in list order.
fn describe(infos: &[wgpu::AdapterInfo], positions: impl Iterator<Item = usize>) -> String {
    let entries: Vec<String> = positions
        .map(|position| {
            let info = &infos[position];
            let mut details = format!("{:?}, {:?}", info.backend, info.device_type);
            if !info.device_pci_bus_id.is_empty() {
                details += &format!(", {}", info.device_pci_bus_id);
            }
            format!("{}. {} ({details})", position + 1, info.name)
        })
        .collect();
    if entries.is_empty() {
        "none".to_owned()
    } else {
        entries.join("; ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(
        name: &str,
        backend: wgpu::Backend,
        device_type: wgpu::DeviceType,
        pci: &str,
    ) -> wgpu::AdapterInfo {
        let mut info = wgpu::AdapterInfo::new(device_type, backend);
        info.name = name.to_owned();
        info.device_pci_bus_id = pci.to_owned();
        info
    }

    fn select(name: &str, pci: &str, index: u32) -> AdapterSelection {
        AdapterSelection {
            name: name.to_owned(),
            pci_bus_id: (!pci.is_empty()).then(|| PciAddress::parse(pci).unwrap()),
            index,
        }
    }

    /// Two identical cards and an integrated GPU, each on Vulkan and on
    /// DX12, which gives both cards the first card's address.
    fn machine() -> Vec<wgpu::AdapterInfo> {
        use wgpu::{Backend::*, DeviceType::*};
        vec![
            info("Intel UHD", Vulkan, IntegratedGpu, "0000:00:02.0"),
            info("RTX 4090", Vulkan, DiscreteGpu, "0000:01:00.0"),
            info("RTX 4090", Vulkan, DiscreteGpu, "0000:02:00.0"),
            info("RTX 4090", Dx12, DiscreteGpu, "0000:01:00.0"),
            info("RTX 4090", Dx12, DiscreteGpu, "0000:01:00.0"),
            info("Intel UHD", Dx12, IntegratedGpu, "0000:00:02.0"),
        ]
    }

    #[test]
    fn pci_addresses_parse_in_the_usual_forms() {
        let full = PciAddress::parse("0000:01:00.0").unwrap();
        assert_eq!(PciAddress::parse("00000000:01:00.0"), Some(full));
        assert_eq!(PciAddress::parse(" 01:00.0 "), Some(full));
        assert_eq!(full.to_string(), "0000:01:00.0");
        let any_function = PciAddress::parse("0000:3B:00").unwrap();
        assert!(any_function.matches(&PciAddress::parse("0000:3b:00.1").unwrap()));
        assert!(!full.matches(&PciAddress::parse("0000:01:00.1").unwrap()));
        for bad in [
            "",
            "1",
            "0000:01:00.",
            "g0:01:00.0",
            "0:0:0:0",
            "+1:00.0",
            "1:2:3:4.0",
        ] {
            assert_eq!(PciAddress::parse(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn the_default_is_the_first_discrete_gpu() {
        assert_eq!(default_position(&machine()), Some(1));
        assert_eq!(default_position(&machine()[..1]), Some(0));
        assert_eq!(default_position(&[]), None);
    }

    #[test]
    fn names_match_whole_and_regardless_of_case() {
        let infos = machine();
        assert_eq!(resolve(&infos, &select(" intel uhd ", "", 0)), Ok(0));
        let error = resolve(&infos, &select("Intel", "", 0)).unwrap_err();
        assert!(matches!(&error, Error::GpuUnavailable(m) if m.contains("1. Intel UHD")));
    }

    #[test]
    fn identical_cards_need_an_index_or_an_address() {
        let infos = machine();
        let error = resolve(&infos, &select("RTX 4090", "", 0)).unwrap_err();
        assert!(
            matches!(&error, Error::InvalidArgument(m) if m.starts_with("2 Vulkan adapters")),
            "{error:?}"
        );
        assert_eq!(resolve(&infos, &select("rtx 4090", "", 2)), Ok(2));
        assert_eq!(resolve(&infos, &select("RTX 4090", "", 3)), Ok(3));
        assert_eq!(resolve(&infos, &select("", "0000:02:00.0", 0)), Ok(2));
        assert_eq!(resolve(&infos, &select("", "02:00", 0)), Ok(2));
        // Vulkan has one card at 01:00.0; DX12's two are not reached.
        assert_eq!(resolve(&infos, &select("", "0000:01:00.0", 0)), Ok(1));
    }

    #[test]
    fn an_index_alone_counts_the_whole_list() {
        let infos = machine();
        assert_eq!(resolve(&infos, &select("", "", 1)), Ok(0));
        assert_eq!(resolve(&infos, &select("", "", 6)), Ok(5));
        assert!(matches!(
            resolve(&infos, &select("", "", 7)),
            Err(Error::GpuUnavailable(_))
        ));
    }

    #[test]
    fn a_missing_adapter_is_reported_with_the_list() {
        let error = resolve(&machine(), &select("Radeon", "", 0)).unwrap_err();
        let Error::GpuUnavailable(message) = error else {
            panic!("{error:?}")
        };
        assert!(message.contains("name \"Radeon\""), "{message}");
        assert!(message.contains("4. RTX 4090 (Dx12, DiscreteGpu, 0000:01:00.0)"));
        let error = resolve(&[], &select("", "0000:01:00.0", 0)).unwrap_err();
        assert!(matches!(&error, Error::GpuUnavailable(m) if m.ends_with("are none")));
        // A backend that reports no address matches no address.
        let metal = wgpu::Backend::Metal;
        let infos = [info("Apple M2", metal, wgpu::DeviceType::IntegratedGpu, "")];
        assert!(resolve(&infos, &select("", "0000:00:00.0", 0)).is_err());
        assert_eq!(resolve(&infos, &select("apple m2", "", 0)), Ok(0));
    }
}
