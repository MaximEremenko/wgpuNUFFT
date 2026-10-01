"""
    WgpuNufft

Nonuniform fast Fourier transforms of types 1, 2 and 3 on the GPU (Vulkan,
DX12, Metal) or the CPU, through the C interface of wgpu-nufft.

    type 1: f[k] = Σⱼ c[j] exp(i·iflag·k⋅x[j])      points to modes
    type 2: c[j] = Σₖ f[k] exp(i·iflag·k⋅x[j])      modes to points
    type 3: f[k] = Σⱼ c[j] exp(i·iflag·s[k]⋅x[j])   points to frequencies

One-call functions `nufft1d1` to `nufft3d3` keep their plans between calls;
[`Plan`](@ref) sets its points once and runs batches. `Float64` arrays run
in native f64 on GPUs that support it and in Df64 (about 44-48 bits) on
others; `Float32` arrays run in f32. Modes are in centered order, frequency
`-floor(n/2)` first along each axis. Type-1 and type-2 coordinates are
radians in [-3π, 3π].

The package loads the shared C library from `ENV["WGPU_NUFFT_C_LIBRARY"]`,
or from `target/release` of the wgpuNUFFT clone it lives in; build it with
`cargo build --release -p wgpu-nufft-c`.
"""
module WgpuNufft

using Libdl

export nufft1d1, nufft1d2, nufft1d3, nufft2d1, nufft2d2, nufft2d3
export nufft3d1, nufft3d2, nufft3d3
export Plan, setpts!, execute, execute!, destroy!, WgpuNufftError

const Real32or64 = Union{Float32,Float64}

# --------------------------------------------------------------------------
# The C library

const LIBRARY = Ref{Ptr{Cvoid}}(C_NULL)
const SYMBOLS = Dict{Symbol,Ptr{Cvoid}}()
const SYMBOLS_LOCK = ReentrantLock()

function library_path()
    path = get(ENV, "WGPU_NUFFT_C_LIBRARY", "")
    isempty(path) || return path
    name = Sys.iswindows() ? "wgpu_nufft_c.dll" :
           Sys.isapple() ? "libwgpu_nufft_c.dylib" : "libwgpu_nufft_c.so"
    return normpath(joinpath(@__DIR__, "..", "..", "..", "target", "release", name))
end

function symbol(name::Symbol)
    lock(SYMBOLS_LOCK) do
        get!(SYMBOLS, name) do
            if LIBRARY[] == C_NULL
                path = library_path()
                isfile(path) || error("wgpu-nufft C library not found at $path; build it " *
                                      "with `cargo build --release -p wgpu-nufft-c` or set " *
                                      "ENV[\"WGPU_NUFFT_C_LIBRARY\"]")
                LIBRARY[] = Libdl.dlopen(path)
            end
            Libdl.dlsym(LIBRARY[], name)
        end
    end
end

prefixed(::Type{Float64}, name) = Symbol("wgpu_nufft", name)
prefixed(::Type{Float32}, name) = Symbol("wgpu_nufftf", name)

"""
    WgpuNufftError(code, message)

An error the library reported: an invalid argument (1), a rejected plan (2),
no usable GPU (3), a GPU error (4) or an internal error (5).
"""
struct WgpuNufftError <: Exception
    code::Int
    message::String
end

Base.showerror(io::IO, e::WgpuNufftError) = print(io, "WgpuNufftError ", e.code, ": ", e.message)

function check(code::Cint)
    if code != 0
        message = unsafe_string(ccall(symbol(:wgpu_nufft_last_error), Cstring, ()))
        throw(WgpuNufftError(code, message))
    end
    return nothing
end

"""The library version."""
version() = unsafe_string(ccall(symbol(:wgpu_nufft_version), Cstring, ()))

"""The GPU plans run on, such as `"NAME (Vulkan)"`, or `""` without one."""
gpu_name() = unsafe_string(ccall(symbol(:wgpu_nufft_gpu_name), Cstring, ()))

"""Releases the shared GPU device and the plans the one-call functions keep."""
shutdown() = (ccall(symbol(:wgpu_nufft_shutdown), Cvoid, ()); nothing)

# --------------------------------------------------------------------------
# Options

"""`wgpu_nufft_opts` of the C interface."""
struct Opts
    backend::Int32
    precision::Int32
    mode_order::Int32
    threads::Int32
    sigma::Float64
end

const BACKENDS = (auto = 0, gpu = 1, cpu = 2)
const PRECISIONS = (auto = 0, f64 = 1, df64 = 2, f32 = 3)

"""
Options, as keywords of every transform and of `Plan`:

- `backend`: `:auto` (the GPU when there is one, default), `:gpu` or `:cpu`
- `precision`: `:auto` (default: native f64 on GPUs that have it, Df64 on
  others, f64 on the CPU, and f32 for `Float32` arrays), `:f64`, `:df64`
  or `:f32`
- `modeord`: 0 for centered modes (default), 1 for FFT order
- `threads`: CPU threads, 0 for all (default)
- `sigma`: upsampling factor, 0 for the default 2
"""
function Opts(; backend::Symbol = :auto, precision::Symbol = :auto, modeord::Integer = 0,
              threads::Integer = 0, sigma::Real = 0.0)
    haskey(BACKENDS, backend) || throw(ArgumentError("backend must be one of $(keys(BACKENDS))"))
    haskey(PRECISIONS, precision) ||
        throw(ArgumentError("precision must be one of $(keys(PRECISIONS))"))
    return Opts(BACKENDS[backend], PRECISIONS[precision], modeord, threads, sigma)
end

# --------------------------------------------------------------------------
# Plans

"""
    Plan(type, n_modes, iflag, ntrans=1, tol=1e-6; dtype=Float64, kwargs...)

A reusable plan of type 1, 2 or 3. `n_modes` holds the modes per axis for
types 1 and 2 and the dimension for type 3; `ntrans` transforms share each
point set. `dtype` (`Float64` or `Float32`) is the real type of the arrays.
The keywords are the options of [`Opts`](@ref). Set the points with
[`setpts!`](@ref), then run with [`execute`](@ref).
"""
mutable struct Plan{T<:Real32or64}
    handle::Ptr{Cvoid}
    # Looked up at creation: finalizers must not take the symbol lock.
    destroy::Ptr{Cvoid}
    type::Int
    n_modes::Vector{Int}
    dim::Int
    ntrans::Int
    M::Int
    N::Int
end

function Plan(type::Integer, n_modes, iflag::Integer, ntrans::Integer = 1, tol::Real = 1e-6;
              dtype::Type{T} = Float64, kwargs...) where {T<:Real32or64}
    opts = Opts(; kwargs...)
    if type == 3
        dim = Int(only(n_modes))
        modes = Int[]
    else
        modes = collect(Int, n_modes)
        dim = length(modes)
    end
    sizes = Int64.(type == 3 ? zeros(Int, dim) : modes)
    handle = Ref{Ptr{Cvoid}}(C_NULL)
    check(ccall(symbol(prefixed(T, :_makeplan)), Cint,
                (Int32, Int32, Ptr{Int64}, Int32, Int64, Float64, Ref{Ptr{Cvoid}}, Ref{Opts}),
                type, dim, sizes, iflag, ntrans, tol, handle, opts))
    plan = Plan{T}(handle[], symbol(prefixed(T, :_destroy)), type, modes, dim, ntrans, 0, 0)
    finalizer(destroy!, plan)
    return plan
end

"""
    destroy!(plan)

Releases the plan now rather than when it is garbage collected.
"""
function destroy!(plan::Plan{T}) where {T}
    if plan.handle != C_NULL
        ccall(plan.destroy, Cvoid, (Ptr{Cvoid},), plan.handle)
        plan.handle = C_NULL
    end
    return nothing
end

"""
    backend(plan) -> :gpu or :cpu
    arithmetic(plan) -> :f64, :df64 or :f32

What a plan runs on.
"""
function backend(plan::Plan{T}) where {T}
    code = ccall(symbol(prefixed(T, :_plan_backend)), Cint, (Ptr{Cvoid},), live(plan))
    return code == 1 ? :gpu : :cpu
end

function arithmetic(plan::Plan{T}) where {T}
    code = ccall(symbol(prefixed(T, :_plan_precision)), Cint, (Ptr{Cvoid},), live(plan))
    return (:auto, :f64, :df64, :f32)[code+1]
end

function live(plan::Plan)
    plan.handle == C_NULL && throw(ArgumentError("the plan was destroyed"))
    return plan.handle
end

contiguous(::Type{T}, values) where {T} = values isa Array{T} ? values : convert(Array{T}, values)

"""
    setpts!(plan, x[, y[, z]]; s=nothing, t=nothing, u=nothing)
    setpts!(plan, X; S=nothing)

Sets the points, one coordinate vector per axis, and for type 3 the target
frequencies `s`, `t`, `u`. A `dim × M` matrix `X` (and `dim × N` matrix `S`)
holds the points of any dimension, one column per point.
"""
function setpts!(plan::Plan{T}, x::AbstractVector, y = nothing, z = nothing;
                 s = nothing, t = nothing, u = nothing) where {T}
    plan.dim <= 3 ||
        throw(ArgumentError("a $(plan.dim)D plan takes its points as a dim × M matrix: setpts!(plan, X)"))
    axes = (x, y, z)[1:plan.dim]
    any(isnothing, axes) && throw(ArgumentError("a $(plan.dim)D plan needs $(plan.dim) coordinate vectors"))
    coords = map(a -> contiguous(T, vec(a)), axes)
    M = length(coords[1])
    all(c -> length(c) == M, coords) || throw(DimensionMismatch("the coordinate vectors differ in length"))
    targets = ()
    N = 0
    if plan.type == 3
        frequencies = (s, t, u)[1:plan.dim]
        any(isnothing, frequencies) && throw(ArgumentError("type 3 needs the target frequencies"))
        targets = map(a -> contiguous(T, vec(a)), frequencies)
        N = length(targets[1])
        all(c -> length(c) == N, targets) ||
            throw(DimensionMismatch("the target frequency vectors differ in length"))
    end
    pointer_or_null(values, axis) = axis <= length(values) ? values[axis] : C_NULL
    GC.@preserve coords targets begin
        check(ccall(symbol(prefixed(T, :_setpts)), Cint,
                    (Ptr{Cvoid}, Int64, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Int64, Ptr{Cvoid},
                     Ptr{Cvoid}, Ptr{Cvoid}),
                    live(plan), M, pointer_or_null(coords, 1), pointer_or_null(coords, 2),
                    pointer_or_null(coords, 3), N, pointer_or_null(targets, 1),
                    pointer_or_null(targets, 2), pointer_or_null(targets, 3)))
    end
    plan.M, plan.N = M, N
    return plan
end

function setpts!(plan::Plan{T}, X::AbstractMatrix; S = nothing) where {T}
    size(X, 1) == plan.dim || throw(DimensionMismatch("X must be $(plan.dim) × M"))
    points = contiguous(T, X)
    targets = plan.type == 3 ? contiguous(T, S) : T[]
    plan.type == 3 && size(targets, 1) != plan.dim &&
        throw(DimensionMismatch("S must be $(plan.dim) × N"))
    M = size(points, 2)
    N = plan.type == 3 ? size(targets, 2) : 0
    check(ccall(symbol(prefixed(T, :_setpts_nd)), Cint,
                (Ptr{Cvoid}, Int64, Ptr{Cvoid}, Int64, Ptr{Cvoid}),
                live(plan), M, points, N, targets))
    plan.M, plan.N = M, N
    return plan
end

"""
    execute(plan, input) -> output
    execute!(plan, input, output)

Runs `ntrans` transforms on the points last set. Type 1 takes the strengths
(`M` values per transform) and returns the modes, an array of size
`(n_modes..., ntrans)`; type 2 takes the modes and returns `M` values per
transform; type 3 takes the strengths and returns `N` values per transform.
With one transform the transform dimension is dropped.
"""
function execute(plan::Plan{T}, input::AbstractArray) where {T}
    dims = plan.type == 1 ? (plan.n_modes..., plan.ntrans) :
           plan.type == 2 ? (plan.M, plan.ntrans) : (plan.N, plan.ntrans)
    if plan.ntrans == 1
        dims = dims[1:end-1]
    end
    output = Array{Complex{T}}(undef, dims...)
    execute!(plan, input, output)
    return output
end

function execute!(plan::Plan{T}, input::AbstractArray, output::AbstractArray) where {T}
    handle = live(plan)
    count_in = plan.type == 2 ? prod(plan.n_modes) : plan.M
    count_out = plan.type == 1 ? prod(plan.n_modes) : plan.type == 2 ? plan.M : plan.N
    length(input) == count_in * plan.ntrans ||
        throw(DimensionMismatch("the input must hold $(count_in * plan.ntrans) values"))
    length(output) == count_out * plan.ntrans ||
        throw(DimensionMismatch("the output must hold $(count_out * plan.ntrans) values"))
    output isa Array{Complex{T}} || throw(ArgumentError("the output must be an Array{Complex{$T}}"))
    source = contiguous(Complex{T}, input)
    # Type 2 reads f and writes c; types 1 and 3 read c and write f.
    c, f = plan.type == 2 ? (output, source) : (source, output)
    check(ccall(symbol(prefixed(T, :_execute)), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
                handle, c, f))
    return output
end

# --------------------------------------------------------------------------
# One-call transforms

floattype(x::AbstractArray{T}) where {T<:Real32or64} = T
floattype(x) = throw(ArgumentError("the coordinates must be Float64 or Float32 arrays"))

function coords(::Type{T}, arrays...) where {T}
    values = map(a -> contiguous(T, vec(a)), arrays)
    length(unique(length.(values))) == 1 ||
        throw(DimensionMismatch("the coordinate vectors differ in length"))
    return values
end

# The C one-call function of `name`. Its arguments are the points, c, the
# sign, the tolerance, the mode counts and f: type 1 reads c and writes f,
# type 2 reads f and writes c.
function call_type12(::Type{T}, kind, name, M, axes, input, iflag, tol, modes, output,
                     opts) where {T}
    arguments = (Int64, ntuple(_ -> Ptr{Cvoid}, length(axes))..., Ptr{Cvoid}, Int32, Float64,
                 ntuple(_ -> Int64, length(modes))..., Ptr{Cvoid}, Ref{Opts})
    c, f = kind == 2 ? (output, input) : (input, output)
    GC.@preserve axes c f begin
        pointers = map(a -> pointer(a), axes)
        check(dynamic_call(symbol(prefixed(T, name)), arguments,
                           (Int64(M), pointers..., Ptr{Cvoid}(pointer(c)), Int32(iflag),
                            Float64(tol), Int64.(modes)..., Ptr{Cvoid}(pointer(f)), opts)))
    end
end

# ccall needs literal argument types, so the one-call functions dispatch on
# their arity through these methods.
@generated function dynamic_call(fptr::Ptr{Cvoid}, ::Type{A}, values::Tuple) where {A<:Tuple}
    types = Tuple(A.parameters)
    arguments = [:(values[$i]) for i in 1:length(types)]
    return :(ccall(fptr, Cint, ($(types...),), $(arguments...)))
end

dynamic_call(fptr, arguments::Tuple, values::Tuple) = dynamic_call(fptr, Tuple{arguments...}, values)

function type12(kind, iflag, tol, modes, axes, input, output_dims, opts)
    T = floattype(axes[1])
    points = coords(T, axes...)
    M = length(points[1])
    source = contiguous(Complex{T}, input)
    output = Array{Complex{T}}(undef, output_dims...)
    name = Symbol(length(axes), "d", kind)
    call_type12(T, kind, name, M, points, source, iflag, tol, modes, output, opts)
    return output
end

function type3(iflag, tol, axes, input, frequencies, opts)
    T = floattype(axes[1])
    points = coords(T, axes...)
    targets = coords(T, frequencies...)
    M, N = length(points[1]), length(targets[1])
    source = contiguous(Complex{T}, input)
    length(source) == M || throw(DimensionMismatch("c must hold one value per point"))
    output = Array{Complex{T}}(undef, N)
    dim = length(axes)
    arguments = (Int64, ntuple(_ -> Ptr{Cvoid}, dim)..., Ptr{Cvoid}, Int32, Float64, Int64,
                 ntuple(_ -> Ptr{Cvoid}, dim)..., Ptr{Cvoid}, Ref{Opts})
    GC.@preserve points targets source output begin
        check(dynamic_call(symbol(prefixed(T, Symbol(dim, "d3"))), arguments,
                           (Int64(M), map(pointer, points)..., Ptr{Cvoid}(pointer(source)),
                            Int32(iflag), Float64(tol), Int64(N), map(pointer, targets)...,
                            Ptr{Cvoid}(pointer(output)), opts)))
    end
    return output
end

"""
    nufft1d1(x, c, iflag, tol, ms; kwargs...) -> f

Type-1 NUFFT in 1D: `ms` modes from the strengths `c` at the points `x`.
The keywords are the options of [`Opts`](@ref); see the module docs for the
definitions. Plans are kept between calls with the same sizes.
"""
function nufft1d1(x, c, iflag, tol, ms; kwargs...)
    length(c) == length(x) || throw(DimensionMismatch("c must hold one value per point"))
    return type12(1, iflag, tol, (ms,), (x,), c, (ms,), Opts(; kwargs...))
end

"""
    nufft1d2(x, iflag, tol, f; kwargs...) -> c

Type-2 NUFFT in 1D: the values at the points `x` of the modes `f`.
"""
nufft1d2(x, iflag, tol, f; kwargs...) =
    type12(2, iflag, tol, (length(f),), (x,), f, (length(x),), Opts(; kwargs...))

"""
    nufft1d3(x, c, iflag, tol, s; kwargs...) -> f

Type-3 NUFFT in 1D: the values at the frequencies `s`.
"""
nufft1d3(x, c, iflag, tol, s; kwargs...) = type3(iflag, tol, (x,), c, (s,), Opts(; kwargs...))

"""
    nufft2d1(x, y, c, iflag, tol, ms, mt; kwargs...) -> f (ms × mt)
"""
function nufft2d1(x, y, c, iflag, tol, ms, mt; kwargs...)
    length(c) == length(x) || throw(DimensionMismatch("c must hold one value per point"))
    return type12(1, iflag, tol, (ms, mt), (x, y), c, (ms, mt), Opts(; kwargs...))
end

"""
    nufft2d2(x, y, iflag, tol, f; kwargs...) -> c, with `f` of size ms × mt
"""
nufft2d2(x, y, iflag, tol, f::AbstractMatrix; kwargs...) =
    type12(2, iflag, tol, size(f), (x, y), f, (length(x),), Opts(; kwargs...))

"""
    nufft2d3(x, y, c, iflag, tol, s, t; kwargs...) -> f
"""
nufft2d3(x, y, c, iflag, tol, s, t; kwargs...) =
    type3(iflag, tol, (x, y), c, (s, t), Opts(; kwargs...))

"""
    nufft3d1(x, y, z, c, iflag, tol, ms, mt, mu; kwargs...) -> f (ms × mt × mu)
"""
function nufft3d1(x, y, z, c, iflag, tol, ms, mt, mu; kwargs...)
    length(c) == length(x) || throw(DimensionMismatch("c must hold one value per point"))
    return type12(1, iflag, tol, (ms, mt, mu), (x, y, z), c, (ms, mt, mu), Opts(; kwargs...))
end

"""
    nufft3d2(x, y, z, iflag, tol, f; kwargs...) -> c, with `f` of size ms × mt × mu
"""
nufft3d2(x, y, z, iflag, tol, f::AbstractArray{<:Any,3}; kwargs...) =
    type12(2, iflag, tol, size(f), (x, y, z), f, (length(x),), Opts(; kwargs...))

"""
    nufft3d3(x, y, z, c, iflag, tol, s, t, u; kwargs...) -> f
"""
nufft3d3(x, y, z, c, iflag, tol, s, t, u; kwargs...) =
    type3(iflag, tol, (x, y, z), c, (s, t, u), Opts(; kwargs...))

end # module
