# Checks the Julia interface against direct sums, on the backend
# ENV["WGPU_NUFFT_TEST_BACKEND"] names: "cpu" (default) or "gpu".
using LinearAlgebra
using Test
using WgpuNufft

const BACKEND = Symbol(get(ENV, "WGPU_NUFFT_TEST_BACKEND", "cpu"))
const TOL = BACKEND == :cpu ? 1e-12 : 1e-9

relerr(a, b) = norm(vec(a) - vec(b)) / norm(vec(b))

# Centered frequencies of each axis, as Cartesian indices of the modes.
freqs(n) = (0:n-1) .- n ÷ 2

function direct1(c, coords, iflag, n_modes)
    f = zeros(ComplexF64, n_modes...)
    for index in CartesianIndices(f), j in eachindex(c)
        phase = sum(freqs(n_modes[a])[index[a]] * coords[a][j] for a in eachindex(n_modes))
        f[index] += c[j] * cis(iflag * phase)
    end
    return f
end

function direct2(f, coords, iflag)
    n_modes = size(f)
    return [sum(f[index] * cis(iflag * sum(freqs(n_modes[a])[index[a]] * coords[a][j]
                                          for a in eachindex(n_modes)))
                for index in CartesianIndices(f)) for j in eachindex(coords[1])]
end

direct3(c, coords, iflag, targets) =
    [sum(c[j] * cis(iflag * sum(targets[a][k] * coords[a][j] for a in eachindex(coords)))
         for j in eachindex(c)) for k in eachindex(targets[1])]

@testset "WgpuNufft $(WgpuNufft.version()) on $BACKEND" begin
    BACKEND == :gpu && @info "GPU: $(WgpuNufft.gpu_name())"
    M = 70
    rand_in(scale, n) = scale .* (2 .* rand(n) .- 1)
    x, y, z = rand_in(π, M), rand_in(π, M), rand_in(π, M)
    c = complex.(randn(M), randn(M))
    ms, mt, mu = 13, 8, 5

    @testset "one-call transforms" begin
        f = nufft1d1(x, c, -1, TOL, ms; backend = BACKEND)
        @test relerr(f, direct1(c, (x,), -1, (ms,))) < 10TOL

        f = nufft2d1(x, y, c, 1, TOL, ms, mt; backend = BACKEND)
        @test size(f) == (ms, mt)
        @test relerr(f, direct1(c, (x, y), 1, (ms, mt))) < 10TOL

        modes = complex.(randn(ms, mt, mu), randn(ms, mt, mu))
        saved = copy(modes)
        values = nufft3d2(x, y, z, 1, TOL, modes; backend = BACKEND)
        @test modes == saved
        @test relerr(values, direct2(modes, (x, y, z), 1)) < 10TOL
        values = nufft1d2(x, -1, TOL, modes[:, 1, 1]; backend = BACKEND)
        @test relerr(values, direct2(modes[:, 1, 1], (x,), -1)) < 10TOL

        N = 40
        s, t = rand_in(9, N), rand_in(6, N)
        f = nufft2d3(x, y, c, 1, TOL, s, t; backend = BACKEND)
        @test relerr(f, direct3(c, (x, y), 1, (s, t))) < 10TOL
    end

    @testset "plan with a batch in FFT order" begin
        plan = Plan(1, (ms, mt), -1, 3, TOL; backend = BACKEND, modeord = 1)
        @test WgpuNufft.backend(plan) == BACKEND
        for round in 1:2
            xr, yr = rand_in(3π, M), rand_in(π, M)
            setpts!(plan, xr, yr)
            strengths = complex.(randn(M, 3), randn(M, 3))
            out = execute(plan, strengths)
            @test size(out) == (ms, mt, 3)
            for k in 1:3
                expected = circshift(direct1(strengths[:, k], (xr, yr), -1, (ms, mt)),
                                     (-(ms ÷ 2), -(mt ÷ 2)))
                @test relerr(out[:, :, k], expected) < 10TOL
            end
        end
        destroy!(plan)
        @test_throws ArgumentError execute(plan, zeros(ComplexF64, M))
    end

    @testset "points of more than three dimensions" begin
        plan = Plan(1, (6, 5, 4, 4), 1, 1, TOL; backend = BACKEND)
        X = rand_in(π, 4 * 30)
        X = reshape(X, 4, 30)
        cs = complex.(randn(30), randn(30))
        setpts!(plan, X)
        f = execute(plan, cs)
        coords = Tuple(X[a, :] for a in 1:4)
        @test relerr(f, direct1(cs, coords, 1, (6, 5, 4, 4))) < 10TOL
    end

    @testset "single precision" begin
        f32 = nufft1d1(Float32.(x), ComplexF32.(c), 1, 1e-5, ms; backend = BACKEND)
        @test eltype(f32) == ComplexF32
        @test relerr(f32, direct1(ComplexF64.(ComplexF32.(c)), (Float64.(Float32.(x)),), 1, (ms,))) < 1e-4
    end

    @testset "errors" begin
        error = try
            nufft1d1([10.0], [1.0 + 0im], 1, 1e-6, 8; backend = BACKEND)
            nothing
        catch e
            e
        end
        @test error isa WgpuNufftError
        @test error.code == 1
        @test occursin("3*pi", error.message)
        @test_throws WgpuNufftError Plan(1, (0,), 1)
    end
    WgpuNufft.shutdown()
end
