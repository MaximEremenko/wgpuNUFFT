//! Rank- and precision-generic WGSL building blocks of the rank-generic GPU
//! paths ([`crate::gpu_nd_bins`], [`crate::gpu_nd_spread`] and the
//! rank-generic type-1 and type-2 plans).
//!
//! Every axis gets its own constants and fold function, so generated shaders
//! have no runtime rank. `F32` and `Df64` plans fold positions in df64 and
//! `F64` plans in native `f64`, like the fixed-rank shaders. `F32` kernel
//! weights come from the `exp` formula, `F64` and `Df64` weights from the
//! host-fitted Horner table, which is also what their deconvolution uses.

use std::fmt::Write as _;

use wgpu_fft::math::DoubleFloat;
use wgpu_fft::FftPrecision;

use crate::gpu_nd::{format_wgsl_f32, format_wgsl_f64};
use crate::kernel::EsKernel;

/// Precision-dependent WGSL types and expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NdWgsl {
    precision: FftPrecision,
}

impl NdWgsl {
    pub(crate) const fn new(precision: FftPrecision) -> Self {
        Self { precision }
    }

    pub(crate) const fn precision(self) -> FftPrecision {
        self.precision
    }

    /// Storage type of one coordinate.
    pub(crate) const fn coordinate_type(self) -> &'static str {
        match self.precision {
            FftPrecision::F32 => "f32",
            FftPrecision::F64 => "f64",
            FftPrecision::Df64 => "vec2<f32>",
        }
    }

    /// Storage type of one complex value.
    pub(crate) const fn complex_type(self) -> &'static str {
        match self.precision {
            FftPrecision::F32 => "vec2<f32>",
            FftPrecision::F64 => "vec2<f64>",
            FftPrecision::Df64 => "vec4<f32>",
        }
    }

    pub(crate) const fn complex_zero(self) -> &'static str {
        match self.precision {
            FftPrecision::F32 => "vec2<f32>(0.0, 0.0)",
            FftPrecision::F64 => "vec2<f64>(0.0lf, 0.0lf)",
            FftPrecision::Df64 => "vec4<f32>(0.0, 0.0, 0.0, 0.0)",
        }
    }

    /// Storage type of one prepared offset `start - position`.
    pub(crate) const fn offset_type(self) -> &'static str {
        match self.precision {
            FftPrecision::F32 | FftPrecision::Df64 => "vec2<f32>",
            FftPrecision::F64 => "f64",
        }
    }

    /// Bytes of one prepared offset: a high/low `f32` pair or an `f64`.
    pub(crate) const fn offset_bytes(self) -> u64 {
        8
    }

    /// Type of a kernel weight in shader code and workgroup memory.
    pub(crate) const fn weight_type(self) -> &'static str {
        match self.precision {
            FftPrecision::F32 => "f32",
            FftPrecision::F64 => "f64",
            FftPrecision::Df64 => "Df64",
        }
    }

    /// Bytes of one kernel weight in workgroup memory.
    pub(crate) const fn weight_bytes(self) -> usize {
        match self.precision {
            FftPrecision::F32 => 4,
            FftPrecision::F64 | FftPrecision::Df64 => 8,
        }
    }

    /// Bytes of one complex value.
    pub(crate) const fn complex_bytes(self) -> usize {
        match self.precision {
            FftPrecision::F32 => 8,
            FftPrecision::F64 | FftPrecision::Df64 => 16,
        }
    }

    /// Product of two weights.
    pub(crate) fn weight_product(self, left: &str, right: &str) -> String {
        match self.precision {
            FftPrecision::F32 | FftPrecision::F64 => format!("({left} * {right})"),
            FftPrecision::Df64 => format!("df64_mul({left}, {right})"),
        }
    }

    /// A complex value scaled by a weight.
    pub(crate) fn complex_scale(self, value: &str, weight: &str) -> String {
        match self.precision {
            FftPrecision::F32 | FftPrecision::F64 => format!("({value} * {weight})"),
            FftPrecision::Df64 => format!("df64_complex_scale({value}, {weight})"),
        }
    }

    /// The sum of two complex values.
    pub(crate) fn complex_add(self, left: &str, right: &str) -> String {
        match self.precision {
            FftPrecision::F32 | FftPrecision::F64 => format!("({left} + {right})"),
            FftPrecision::Df64 => format!("df64_complex_add({left}, {right})"),
        }
    }

    /// Whether shaders of this precision need the df64 library.
    pub(crate) const fn needs_df64(self) -> bool {
        matches!(self.precision, FftPrecision::F32 | FftPrecision::Df64)
    }

    /// Prepends the df64 library when this precision needs it.
    pub(crate) fn with_library(self, source: &str) -> String {
        if self.needs_df64() {
            format!("{}\n{source}", wgpu_fft::kernels::DF64_WGSL)
        } else {
            source.to_owned()
        }
    }

    /// The folded position of a coordinate expression along `axis`: a `Df64`
    /// for `F32` and `Df64` plans, an `f64` for `F64` plans.
    pub(crate) fn fold(self, axis: usize, coordinate: &str) -> String {
        match self.precision {
            FftPrecision::F32 | FftPrecision::F64 => format!("fold_position_{axis}({coordinate})"),
            FftPrecision::Df64 => {
                format!("fold_position_{axis}(Df64({coordinate}.x, {coordinate}.y))")
            }
        }
    }

    /// The grid cell `floor(position)` of a folded position, clamped onto the
    /// grid so that even a coordinate outside the contract lands in a bin.
    pub(crate) fn cell(self, axis: usize, position: &str) -> String {
        let floor = match self.precision {
            FftPrecision::F32 | FftPrecision::Df64 => format!("floor_df64_to_i32({position})"),
            FftPrecision::F64 => format!("i32(floor({position}))"),
        };
        format!("min(u32(max({floor}, 0)), FINE_{axis} - 1u)")
    }

    /// The support start `ceil(position - w/2)` of a folded position.
    pub(crate) fn start(self, position: &str) -> String {
        match self.precision {
            FftPrecision::F32 | FftPrecision::Df64 => {
                format!("ceil_df64_to_i32(df64_sub({position}, Df64(HALF_WIDTH, 0.0)))")
            }
            FftPrecision::F64 => format!("i32(ceil({position} - HALF_WIDTH))"),
        }
    }

    /// The prepared offset `start - position` in storage form.
    pub(crate) fn offset(self, start: &str, position: &str) -> String {
        match self.precision {
            FftPrecision::F32 | FftPrecision::Df64 => format!(
                "df64_pack(df64_sub(Df64(f32({start}), 0.0), {position}))"
            ),
            FftPrecision::F64 => format!("(f64({start}) - {position})"),
        }
    }

    /// An offset past every kernel support, so every weight is zero.
    pub(crate) const fn outside_offset(self) -> &'static str {
        match self.precision {
            FftPrecision::F32 | FftPrecision::Df64 => "vec2<f32>(1024.0, 0.0)",
            FftPrecision::F64 => "1024.0lf",
        }
    }

    /// Whether a coordinate lies within the fold's reach of two periods.
    /// `F32` and `Df64` test magnitude bits, so NaN and infinities fail even
    /// where the compiler assumes they never occur; `F64` compares.
    pub(crate) fn in_reach(self, coordinate: &str) -> String {
        match self.precision {
            FftPrecision::F32 => {
                format!("((bitcast<u32>({coordinate}) & 0x7fffffffu) <= FOLD_REACH_BITS)")
            }
            FftPrecision::Df64 => {
                format!("((bitcast<u32>({coordinate}.x) & 0x7fffffffu) <= FOLD_REACH_BITS)")
            }
            FftPrecision::F64 => format!("(abs({coordinate}) <= FOLD_REACH)"),
        }
    }
}

/// Per-axis grid constants and fold functions.
///
/// Defines `FINE_{a}` (u32), `FINE_{a}_I32`, `FINE_STRIDE_{a}` and
/// `FINE_COUNT` for every axis, the fold reach (`FOLD_REACH_BITS` or
/// `FOLD_REACH`), `HALF_WIDTH`, and `fold_position_{a}`. `F32` and `Df64`
/// shaders also get `floor_df64_to_i32`, `ceil_df64_to_i32` and `df64_pack`,
/// and need the df64 library in front ([`NdWgsl::with_library`]).
pub(crate) fn position_wgsl(fine_shape: &[usize], kernel: EsKernel, precision: FftPrecision) -> String {
    let mut source = String::new();
    let mut stride = 1usize;
    for (axis, &length) in fine_shape.iter().enumerate() {
        let _ = writeln!(source, "const FINE_{axis}: u32 = {length}u;");
        let _ = writeln!(source, "const FINE_{axis}_I32: i32 = {length}i;");
        let _ = writeln!(source, "const FINE_STRIDE_{axis}: u32 = {stride}u;");
        stride *= length;
    }
    let _ = writeln!(source, "const FINE_COUNT: u32 = {stride}u;");
    match precision {
        FftPrecision::F32 | FftPrecision::Df64 => {
            let _ = writeln!(
                source,
                "const HALF_WIDTH: f32 = {};",
                format_wgsl_f32(kernel.half_width() as f32)
            );
            // Bits of 4*pi, a magnitude the fold's two periods bring onto the grid.
            let _ = writeln!(
                source,
                "const FOLD_REACH_BITS: u32 = {}u;",
                (4.0 * std::f32::consts::PI).to_bits()
            );
            for (axis, &length) in fine_shape.iter().enumerate() {
                let scale = length as f64 / std::f64::consts::TAU;
                let scale_hi = scale as f32;
                let scale_lo = (scale - f64::from(scale_hi)) as f32;
                let _ = writeln!(
                    source,
                    "const FINE_{axis}_F32: f32 = {};",
                    format_wgsl_f32(length as f32)
                );
                let _ = writeln!(
                    source,
                    "const POSITION_SCALE_{axis}_HI: f32 = {};",
                    format_wgsl_f32(scale_hi)
                );
                let _ = writeln!(
                    source,
                    "const POSITION_SCALE_{axis}_LO: f32 = {};",
                    format_wgsl_f32(scale_lo)
                );
                let _ = writeln!(
                    source,
                    "const GRID_ORIGIN_{axis}: f32 = {};",
                    format_wgsl_f32((length / 2) as f32)
                );
            }
            let point_type = if precision == FftPrecision::F32 {
                "f32"
            } else {
                "Df64"
            };
            let scaled = if precision == FftPrecision::F32 {
                "df64_mul(Df64(point, 0.0), Df64(scale_hi, scale_lo))"
            } else {
                "df64_mul(point, Df64(scale_hi, scale_lo))"
            };
            let _ = write!(
                source,
                r#"
fn position_is_negative(value: Df64) -> bool {{
    return value.hi < 0.0 || (value.hi == 0.0 && value.lo < 0.0);
}}

fn position_at_least_grid(value: Df64, fine_length: f32) -> bool {{
    return value.hi > fine_length || (value.hi == fine_length && value.lo >= 0.0);
}}

// A coordinate within two periods of the grid folded into [0, n) in df64.
fn fold_position(point: {point_type}, scale_hi: f32, scale_lo: f32, origin: f32, fine_length: f32) -> Df64 {{
    var position = df64_add({scaled}, Df64(origin, 0.0));
    if (position_is_negative(position)) {{ position = df64_add(position, Df64(fine_length, 0.0)); }}
    if (position_is_negative(position)) {{ position = df64_add(position, Df64(fine_length, 0.0)); }}
    if (position_at_least_grid(position, fine_length)) {{ position = df64_sub(position, Df64(fine_length, 0.0)); }}
    if (position_at_least_grid(position, fine_length)) {{ position = df64_sub(position, Df64(fine_length, 0.0)); }}
    return position;
}}

fn floor_df64_to_i32(value: Df64) -> i32 {{
    let base = floor(value.hi);
    let remainder = df64_sub(value, Df64(base, 0.0));
    let has_negative_remainder = remainder.hi < 0.0 || (remainder.hi == 0.0 && remainder.lo < 0.0);
    return i32(base) - select(0, 1, has_negative_remainder);
}}

fn ceil_df64_to_i32(value: Df64) -> i32 {{
    let base = floor(value.hi);
    let remainder = df64_sub(value, Df64(base, 0.0));
    let has_positive_remainder = remainder.hi > 0.0 || (remainder.hi == 0.0 && remainder.lo > 0.0);
    return i32(base) + select(0, 1, has_positive_remainder);
}}

fn df64_pack(value: Df64) -> vec2<f32> {{
    return vec2<f32>(value.hi, value.lo);
}}
"#
            );
            for axis in 0..fine_shape.len() {
                let _ = write!(
                    source,
                    "
fn fold_position_{axis}(point: {point_type}) -> Df64 {{
    return fold_position(point, POSITION_SCALE_{axis}_HI, POSITION_SCALE_{axis}_LO, GRID_ORIGIN_{axis}, FINE_{axis}_F32);
}}
"
                );
            }
        }
        FftPrecision::F64 => {
            let _ = writeln!(
                source,
                "const HALF_WIDTH: f64 = {};",
                format_wgsl_f64(kernel.half_width())
            );
            let _ = writeln!(
                source,
                "const FOLD_REACH: f64 = {};",
                format_wgsl_f64(4.0 * std::f64::consts::PI)
            );
            for (axis, &length) in fine_shape.iter().enumerate() {
                let _ = writeln!(
                    source,
                    "const FINE_{axis}_F64: f64 = {};",
                    format_wgsl_f64(length as f64)
                );
                let _ = writeln!(
                    source,
                    "const POSITION_SCALE_{axis}: f64 = {};",
                    format_wgsl_f64(length as f64 / std::f64::consts::TAU)
                );
                let _ = writeln!(
                    source,
                    "const GRID_ORIGIN_{axis}: f64 = {};",
                    format_wgsl_f64((length / 2) as f64)
                );
            }
            source.push_str(
                r#"
// A coordinate within two periods of the grid folded into [0, n).
fn fold_position(point: f64, scale: f64, origin: f64, fine_length: f64) -> f64 {
    var position = point * scale + origin;
    if (position < 0.0lf) { position = position + fine_length; }
    if (position < 0.0lf) { position = position + fine_length; }
    if (position >= fine_length) { position = position - fine_length; }
    if (position >= fine_length) { position = position - fine_length; }
    return position;
}
"#,
            );
            for axis in 0..fine_shape.len() {
                let _ = write!(
                    source,
                    "
fn fold_position_{axis}(point: f64) -> f64 {{
    return fold_position(point, POSITION_SCALE_{axis}, GRID_ORIGIN_{axis}, FINE_{axis}_F64);
}}
"
                );
            }
        }
    }
    source
}

/// The kernel weight functions: `support_weight(offset, j)` is the weight of
/// support cell `start + j` for a point with prepared offset
/// `start - position`, of [`NdWgsl::weight_type`].
///
/// Defines `WIDTH`, `WIDTH_I32` and, for `F32`, `WIDTH_F32` and `BETA`.
/// Needs `HALF_WIDTH` from [`position_wgsl`].
pub(crate) fn weight_wgsl(kernel: EsKernel, precision: FftPrecision) -> String {
    let width = kernel.width();
    let mut source = format!("const WIDTH: u32 = {width}u;\nconst WIDTH_I32: i32 = {width}i;\n");
    match precision {
        FftPrecision::F32 => {
            let _ = write!(
                source,
                r#"const WIDTH_F32: f32 = {width}.0;
const BETA: f32 = {beta};

fn es_weight(distance: f32) -> f32 {{
    let scaled = 2.0 * abs(distance) / WIDTH_F32;
    let squared = scaled * scaled;
    if (squared >= 1.0) {{ return 0.0; }}
    return exp(BETA * (sqrt(max(0.0, 1.0 - squared)) - 1.0));
}}

// Distance from support cell `start + j` is `j + (start - position)`, with
// the df64 offset split into high and low f32 words.
fn support_weight(offset: vec2<f32>, j: u32) -> f32 {{
    return es_weight((f32(j) + offset.x) + offset.y);
}}
"#,
                beta = format_wgsl_f32(kernel.beta() as f32),
            );
        }
        FftPrecision::F64 => {
            let table = kernel.horner_table();
            let coefficients = table
                .coefficients()
                .iter()
                .map(|&value| format_wgsl_f64(value))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = write!(
                source,
                r#"const HORNER_COEFFICIENT_COUNT: u32 = {count}u;
const HORNER_COEFFICIENTS: array<f64, {total}> = array<f64, {total}>({coefficients});

fn es_weight(distance: f64) -> f64 {{
    if (abs(distance) >= HALF_WIDTH) {{ return 0.0lf; }}
    let panel = u32(clamp(i32(ceil(distance + HALF_WIDTH)) - 1, 0, WIDTH_I32 - 1));
    let local = 2.0lf * (distance - f64(panel)) + f64(WIDTH_I32 - 1);
    var value = 0.0lf;
    for (var coefficient = 0u; coefficient < HORNER_COEFFICIENT_COUNT; coefficient = coefficient + 1u) {{
        value = value * local + HORNER_COEFFICIENTS[coefficient * WIDTH + panel];
    }}
    return value;
}}

fn support_weight(offset: f64, j: u32) -> f64 {{
    return es_weight(f64(j) + offset);
}}
"#,
                count = table.coefficient_count(),
                total = table.coefficients().len(),
            );
        }
        FftPrecision::Df64 => {
            let table = kernel.horner_table();
            let coefficients = table
                .coefficients()
                .iter()
                .map(|&value| {
                    let value = DoubleFloat::from_f64(value);
                    format!(
                        "Df64({}, {})",
                        format_wgsl_f32(value.hi),
                        format_wgsl_f32(value.lo)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            let half_width = DoubleFloat::from_f64(kernel.half_width());
            let _ = write!(
                source,
                r#"const HORNER_COEFFICIENT_COUNT: u32 = {count}u;
const HORNER_COEFFICIENTS: array<Df64, {total}> = array<Df64, {total}>({coefficients});
const DF64_HALF_WIDTH: Df64 = Df64({half_width_hi}, {half_width_lo});

fn df64_abs(value: Df64) -> Df64 {{
    if (value.hi < 0.0 || (value.hi == 0.0 && value.lo < 0.0)) {{ return df64_neg(value); }}
    return value;
}}

fn df64_at_least(left: Df64, right: Df64) -> bool {{
    return left.hi > right.hi || (left.hi == right.hi && left.lo >= right.lo);
}}

fn es_weight(distance: Df64) -> Df64 {{
    if (df64_at_least(df64_abs(distance), DF64_HALF_WIDTH)) {{ return Df64(0.0, 0.0); }}
    let panel_i32 = clamp(ceil_df64_to_i32(df64_add(distance, DF64_HALF_WIDTH)) - 1, 0, WIDTH_I32 - 1);
    let panel = u32(panel_i32);
    let local = df64_add(
        df64_mul(Df64(2.0, 0.0), df64_sub(distance, Df64(f32(panel_i32), 0.0))),
        Df64(f32(WIDTH_I32 - 1), 0.0),
    );
    var value = Df64(0.0, 0.0);
    for (var coefficient = 0u; coefficient < HORNER_COEFFICIENT_COUNT; coefficient = coefficient + 1u) {{
        value = df64_add(df64_mul(value, local), HORNER_COEFFICIENTS[coefficient * WIDTH + panel]);
    }}
    return value;
}}

fn support_weight(offset: vec2<f32>, j: u32) -> Df64 {{
    return es_weight(df64_add(Df64(f32(j), 0.0), Df64(offset.x, offset.y)));
}}
"#,
                count = table.coefficient_count(),
                total = table.coefficients().len(),
                half_width_hi = format_wgsl_f32(half_width.hi),
                half_width_lo = format_wgsl_f32(half_width.lo),
            );
        }
    }
    source
}

/// `wrap_index(index, length)`: an index within one period of `[0, length)`
/// wrapped into it.
pub(crate) const WRAP_INDEX_WGSL: &str = r#"
fn wrap_index(index: i32, length: i32) -> u32 {
    var wrapped = index;
    if (wrapped < 0) { wrapped = wrapped + length; }
    if (wrapped >= length) { wrapped = wrapped - length; }
    return u32(wrapped);
}
"#;

/// Axis-zero-fastest linear index of per-axis expressions, nested from the
/// last axis inward: `e0 + L0 * (e1 + L1 * (...))`.
pub(crate) fn linear_index(terms: &[String], lengths: &[String]) -> String {
    let last = terms.len() - 1;
    let mut index = terms[last].clone();
    for axis in (0..last).rev() {
        index = format!("{} + {} * ({index})", terms[axis], lengths[axis]);
    }
    index
}

/// Validates a generated shader of `precision` in tests.
#[cfg(test)]
pub(crate) fn assert_valid_nd_wgsl(precision: FftPrecision, source: &str) {
    match precision {
        FftPrecision::F64 => crate::wgsl_validation::assert_valid_wgsl_f64(source),
        FftPrecision::F32 | FftPrecision::Df64 => crate::wgsl_validation::assert_valid_wgsl(source),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(precision: FftPrecision, dimensions: usize) -> String {
        let kernel = EsKernel::for_tolerance(1.0e-6, 2.0).unwrap();
        let shape = (0..dimensions)
            .map(|axis| 14 + 2 * (axis % 2))
            .collect::<Vec<_>>();
        let types = NdWgsl::new(precision);
        let mut body = String::new();
        for axis in 0..dimensions {
            let _ = writeln!(
                body,
                "    let position_{axis} = {};\n    let cell_{axis} = {};\n    let start_{axis} = {};\n    let offset_{axis} = {};\n    let inside_{axis} = {};\n    let weight_{axis} = support_weight(offset_{axis}, 3u);",
                types.fold(axis, &format!("points[{axis}]")),
                types.cell(axis, &format!("position_{axis}")),
                types.start(&format!("position_{axis}")),
                types.offset(&format!("start_{axis}"), &format!("position_{axis}")),
                types.in_reach(&format!("points[{axis}]")),
            );
        }
        let source = format!(
            "{}\n{}\n{WRAP_INDEX_WGSL}\n@group(0) @binding(0) var<storage, read> points: array<{}>;\n\
             @group(0) @binding(1) var<storage, read_write> sink: array<{}>;\n\
             @compute @workgroup_size(1)\nfn main() {{\n{body}    sink[0] = {};\n}}\n",
            position_wgsl(&shape, kernel, precision),
            weight_wgsl(kernel, precision),
            types.coordinate_type(),
            types.complex_type(),
            types.complex_scale("sink[1]", "weight_0"),
        );
        types.with_library(&source)
    }

    #[test]
    fn building_blocks_validate_in_every_precision_and_rank() {
        for precision in [FftPrecision::F32, FftPrecision::F64, FftPrecision::Df64] {
            for dimensions in [1, 4, 8] {
                assert_valid_nd_wgsl(precision, &probe(precision, dimensions));
            }
        }
    }

    #[test]
    fn linear_index_nests_axis_zero_fastest() {
        let terms = ["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let lengths = ["L0".to_owned(), "L1".to_owned(), "L2".to_owned()];
        assert_eq!(linear_index(&terms, &lengths), "a + L0 * (b + L1 * (c))");
        assert_eq!(linear_index(&terms[..1], &lengths[..1]), "a");
    }
}
