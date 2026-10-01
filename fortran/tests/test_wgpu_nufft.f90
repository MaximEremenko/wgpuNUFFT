! Checks the Fortran interface against direct sums.
! Usage: test_wgpu_nufft [cpu|gpu]   (default cpu)
program test_wgpu_nufft
    use, intrinsic :: iso_c_binding
    use wgpu_nufft
    implicit none

    real(c_double), parameter :: pi = 3.14159265358979323846_c_double
    integer(c_int64_t), parameter :: M = 70, N = 45
    type(wgpu_nufft_opts) :: opts
    character(len=16) :: argument
    real(c_double) :: x(M), y(M), z(M), s(N), t(N), u(N), tolerance, eps
    complex(c_double_complex) :: c(M), f1(13), f2(9, 6), f3(N), expected1(13), expected2(9, 6)
    complex(c_double_complex) :: expected3(N), values(M), expected_values(M)
    integer :: ier, failures

    failures = 0
    argument = "cpu"
    if (command_argument_count() >= 1) call get_command_argument(1, argument)
    if (trim(argument) == "gpu") then
        opts%backend = WGPU_NUFFT_BACKEND_GPU
        print "(a, a)", "GPU: ", wgpu_nufft_gpu_name()
        eps = 1.0d-9
        tolerance = 1.0d-8
    else
        opts%backend = WGPU_NUFFT_BACKEND_CPU
        eps = 1.0d-12
        tolerance = 1.0d-10
    end if
    print "(a, a)", "wgpu-nufft ", wgpu_nufft_version()

    call fill(x, pi); call fill(y, pi); call fill(z, pi)
    call fill(s, 9.0_c_double); call fill(t, 6.0_c_double); call fill(u, 4.0_c_double)
    call fill_complex(c)

    ! Type 1 in 1D, isign -1.
    call wgpu_nufft1d1(M, x, c, -1, eps, 13_c_int64_t, f1, ier, opts)
    call expect_success("1d1")
    call direct1d1(-1, expected1)
    call compare("1d1", f1, expected1, size(f1))

    ! Type 2 in 2D, isign +1, from the modes the 2D type 1 makes.
    call direct2d1(1, expected2)
    call wgpu_nufft2d2(M, x, y, values, 1, eps, 9_c_int64_t, 6_c_int64_t, expected2, ier, opts)
    call expect_success("2d2")
    call direct2d2(1, expected2, expected_values)
    call compare("2d2", values, expected_values, int(M))

    ! Type 1 in 2D through the one-call routine, then its arrays in order.
    call wgpu_nufft2d1(M, x, y, c, 1, eps, 9_c_int64_t, 6_c_int64_t, f2, ier, opts)
    call expect_success("2d1")
    call compare("2d1", f2, expected2, size(f2))

    ! Type 3 in 3D.
    call wgpu_nufft3d3(M, x, y, z, c, 1, eps, N, s, t, u, f3, ier, opts)
    call expect_success("3d3")
    call direct3d3(1, expected3)
    call compare("3d3", f3, expected3, int(N))

    call plan_with_batch()
    call single_precision()
    call errors()

    call wgpu_nufft_shutdown()
    if (failures > 0) then
        print "(i0, a)", failures, " check(s) failed"
        error stop 1
    end if
    print "(a)", "all checks passed"

contains

    subroutine fill(values, scale)
        real(c_double), intent(out) :: values(:)
        real(c_double), intent(in) :: scale
        integer, save :: state = 12345
        integer :: i
        do i = 1, size(values)
            state = mod(state * 1103 + 12345, 65536)
            values(i) = (2.0_c_double * state / 65536.0_c_double - 1.0_c_double) * scale
        end do
    end subroutine fill

    subroutine fill_complex(values)
        complex(c_double_complex), intent(out) :: values(:)
        real(c_double) :: re(size(values)), im(size(values))
        call fill(re, 1.0_c_double)
        call fill(im, 1.0_c_double)
        values = cmplx(re, im, c_double_complex)
    end subroutine fill_complex

    subroutine expect_success(name)
        character(len=*), intent(in) :: name
        if (ier /= WGPU_NUFFT_SUCCESS) then
            print "(a, a, i0, a, a)", name, ": error ", ier, ": ", wgpu_nufft_error_message()
            failures = failures + 1
        end if
    end subroutine expect_success

    subroutine compare(name, actual, expected, count)
        character(len=*), intent(in) :: name
        integer, intent(in) :: count
        complex(c_double_complex), intent(in) :: actual(count), expected(count)
        real(c_double) :: error
        error = sqrt(sum(abs(actual - expected)**2) / sum(abs(expected)**2))
        if (error > tolerance) then
            print "(a, a, es10.3)", name, ": relative error ", error
            failures = failures + 1
        else
            print "(a, a, es10.3)", name, ": ok, relative error ", error
        end if
    end subroutine compare

    ! exp(i * sign * angle)
    complex(c_double_complex) function phase(sign, angle)
        integer, intent(in) :: sign
        real(c_double), intent(in) :: angle
        phase = cmplx(cos(angle), sign * sin(angle), c_double_complex)
    end function phase

    subroutine direct1d1(sign, f)
        integer, intent(in) :: sign
        complex(c_double_complex), intent(out) :: f(:)
        integer :: k, j
        f = 0
        do k = 1, size(f)
            do j = 1, int(M)
                f(k) = f(k) + c(j) * phase(sign, (k - 1 - size(f) / 2) * x(j))
            end do
        end do
    end subroutine direct1d1

    subroutine direct2d1(sign, f)
        integer, intent(in) :: sign
        complex(c_double_complex), intent(out) :: f(:, :)
        integer :: k1, k2, j
        f = 0
        do k2 = 1, size(f, 2)
            do k1 = 1, size(f, 1)
                do j = 1, int(M)
                    f(k1, k2) = f(k1, k2) + c(j) * phase(sign, &
                        (k1 - 1 - size(f, 1) / 2) * x(j) + (k2 - 1 - size(f, 2) / 2) * y(j))
                end do
            end do
        end do
    end subroutine direct2d1

    subroutine direct2d2(sign, f, values)
        integer, intent(in) :: sign
        complex(c_double_complex), intent(in) :: f(:, :)
        complex(c_double_complex), intent(out) :: values(:)
        integer :: k1, k2, j
        values = 0
        do j = 1, int(M)
            do k2 = 1, size(f, 2)
                do k1 = 1, size(f, 1)
                    values(j) = values(j) + f(k1, k2) * phase(sign, &
                        (k1 - 1 - size(f, 1) / 2) * x(j) + (k2 - 1 - size(f, 2) / 2) * y(j))
                end do
            end do
        end do
    end subroutine direct2d2

    subroutine direct3d3(sign, f)
        integer, intent(in) :: sign
        complex(c_double_complex), intent(out) :: f(:)
        integer :: k, j
        f = 0
        do k = 1, int(N)
            do j = 1, int(M)
                f(k) = f(k) + c(j) * phase(sign, s(k) * x(j) + t(k) * y(j) + u(k) * z(j))
            end do
        end do
    end subroutine direct3d3

    ! A type-1 plan reused for two point sets, three transforms at a time.
    subroutine plan_with_batch()
        type(wgpu_nufft_plan) :: plan
        complex(c_double_complex) :: strengths(M, 3), modes(13, 3), expected(13)
        complex(c_double_complex) :: saved(M)
        real(c_double) :: points(M)
        integer :: round, transform
        call wgpu_nufft_makeplan(1, 1, [13_c_int64_t], -1, 3, eps, plan, ier, opts)
        call expect_success("makeplan")
        print "(a, i0, a, i0)", "plan backend ", wgpu_nufft_plan_backend(plan), &
            ", precision ", wgpu_nufft_plan_precision(plan)
        saved = c
        do round = 1, 2
            call fill(points, 3.0_c_double * pi)
            call wgpu_nufft_setpts(plan, M, points, ier=ier)
            call expect_success("setpts")
            do transform = 1, 3
                call fill_complex(strengths(:, transform))
            end do
            call wgpu_nufft_execute(plan, strengths, modes, ier)
            call expect_success("execute")
            x = points
            do transform = 1, 3
                c = strengths(:, transform)
                call direct1d1(-1, expected)
                call compare("plan batch", modes(:, transform), expected, 13)
            end do
        end do
        c = saved
        call wgpu_nufft_destroy(plan)
    end subroutine plan_with_batch

    subroutine single_precision()
        real(c_float) :: xf(M)
        complex(c_float_complex) :: cf(M), ff(13)
        complex(c_double_complex) :: expected(13)
        real(c_double) :: saved_tolerance
        xf = real(x, c_float)
        cf = cmplx(c, kind=c_float_complex)
        call wgpu_nufftf1d1(M, xf, cf, 1, 1.0d-5, 13_c_int64_t, ff, ier, opts)
        call expect_success("f1d1")
        x = real(xf, c_double)
        c = cmplx(cf, kind=c_double_complex)
        call direct1d1(1, expected)
        saved_tolerance = tolerance
        tolerance = 1.0d-4
        call compare("f1d1", cmplx(ff, kind=c_double_complex), expected, 13)
        tolerance = saved_tolerance
    end subroutine single_precision

    subroutine errors()
        type(wgpu_nufft_plan) :: plan
        real(c_double) :: far(1)
        call wgpu_nufft_makeplan(1, 1, [0_c_int64_t], 1, 1, 1.0d-6, plan, ier, opts)
        if (ier /= WGPU_NUFFT_ERROR_INVALID_ARGUMENT) then
            print "(a, i0)", "errors: zero modes gave ", ier
            failures = failures + 1
        end if
        call wgpu_nufft_makeplan(1, 1, [8_c_int64_t], 1, 1, 1.0d-6, plan, ier, opts)
        call expect_success("errors makeplan")
        far = 10.0_c_double
        call wgpu_nufft_setpts(plan, 1_c_int64_t, far, ier=ier)
        if (ier /= WGPU_NUFFT_ERROR_INVALID_ARGUMENT .or. index(wgpu_nufft_error_message(), "3*pi") == 0) then
            print "(a, i0, a, a)", "errors: a far point gave ", ier, ": ", wgpu_nufft_error_message()
            failures = failures + 1
        else
            print "(a, a)", "errors: ok, ", wgpu_nufft_error_message()
        end if
        call wgpu_nufft_destroy(plan)
    end subroutine errors

end program test_wgpu_nufft
