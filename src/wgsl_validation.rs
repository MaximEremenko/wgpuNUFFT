//! Test-only validation of generated WGSL.
//!
//! wgpu exposes naga only on native targets, so on wasm32 the checks compile
//! to nothing; browser test runs execute only `wasm_bindgen_test` cases.

/// Parses and validates `source` for devices without `SHADER_F64`.
pub(crate) fn assert_valid_wgsl(source: &str) {
    validate(source, false);
}

/// Parses and validates `source`, allowing `f64`.
pub(crate) fn assert_valid_wgsl_f64(source: &str) {
    validate(source, true);
}

#[cfg(not(target_arch = "wasm32"))]
fn validate(source: &str, float64: bool) {
    let capabilities = if float64 {
        wgpu::naga::valid::Capabilities::FLOAT64
    } else {
        wgpu::naga::valid::Capabilities::empty()
    };
    let module = wgpu::naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), capabilities)
        .validate(&module)
        .unwrap_or_else(|error| panic!("{error:?}"));
}

#[cfg(target_arch = "wasm32")]
fn validate(_source: &str, _float64: bool) {}
