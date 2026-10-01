! Fortran interface to wgpu-nufft, over the C interface of capi/.
!
! Nonuniform fast Fourier transforms of types 1, 2 and 3 on the GPU or the
! CPU. Sizes (M, N, ms, mt, mu, n_modes) are integer(8); every routine sets
! ier to WGPU_NUFFT_SUCCESS (0) or an error code, which
! wgpu_nufft_error_message() then describes. Routines named wgpu_nufft* take
! double-precision arrays and wgpu_nufftf* single-precision ones.
!
! Modes are stored in array order f(ms, mt, mu), dimension one fastest, from
! frequency -floor(n/2) at index 1 (centered order) unless the options select
! FFT order. Batches add a last dimension: c(M, ntrans), f(ms, mt, ntrans).
! Type-1 and type-2 coordinates are radians in [-3*pi, 3*pi].
!
!   type 1: f(k) = sum_j c(j) exp(i * isign * k . x(j))
!   type 2: c(j) = sum_k f(k) exp(i * isign * k . x(j))
!   type 3: f(k) = sum_j c(j) exp(i * isign * s(k) . x(j))
module wgpu_nufft
    use, intrinsic :: iso_c_binding
    implicit none
    private

    integer, parameter, public :: WGPU_NUFFT_SUCCESS = 0
    integer, parameter, public :: WGPU_NUFFT_ERROR_INVALID_ARGUMENT = 1
    integer, parameter, public :: WGPU_NUFFT_ERROR_PLAN = 2
    integer, parameter, public :: WGPU_NUFFT_ERROR_GPU_UNAVAILABLE = 3
    integer, parameter, public :: WGPU_NUFFT_ERROR_GPU = 4
    integer, parameter, public :: WGPU_NUFFT_ERROR_INTERNAL = 5

    integer, parameter, public :: WGPU_NUFFT_BACKEND_AUTO = 0
    integer, parameter, public :: WGPU_NUFFT_BACKEND_GPU = 1
    integer, parameter, public :: WGPU_NUFFT_BACKEND_CPU = 2

    integer, parameter, public :: WGPU_NUFFT_PRECISION_AUTO = 0
    integer, parameter, public :: WGPU_NUFFT_PRECISION_F64 = 1
    integer, parameter, public :: WGPU_NUFFT_PRECISION_DF64 = 2
    integer, parameter, public :: WGPU_NUFFT_PRECISION_F32 = 3

    integer, parameter, public :: WGPU_NUFFT_MODE_ORDER_CENTERED = 0
    integer, parameter, public :: WGPU_NUFFT_MODE_ORDER_FFT = 1

    ! wgpu_nufft_opts of the C interface; the defaults select the GPU when
    ! there is one, its best double arithmetic, centered modes, every CPU
    ! thread and an upsampling factor of 2.
    type, bind(c), public :: wgpu_nufft_opts
        integer(c_int32_t) :: backend = 0
        integer(c_int32_t) :: precision = 0
        integer(c_int32_t) :: mode_order = 0
        integer(c_int32_t) :: threads = 0
        real(c_double) :: sigma = 0.0_c_double
    end type wgpu_nufft_opts

    ! Plans for double- and single-precision arrays.
    type, public :: wgpu_nufft_plan
        type(c_ptr) :: handle = c_null_ptr
    end type wgpu_nufft_plan

    type, public :: wgpu_nufftf_plan
        type(c_ptr) :: handle = c_null_ptr
    end type wgpu_nufftf_plan

    public :: wgpu_nufft_error_message, wgpu_nufft_version, wgpu_nufft_gpu_name
    public :: wgpu_nufft_shutdown
    public :: wgpu_nufft_makeplan, wgpu_nufft_setpts, wgpu_nufft_setpts_nd
    public :: wgpu_nufft_execute, wgpu_nufft_destroy
    public :: wgpu_nufft_plan_backend, wgpu_nufft_plan_precision
    public :: wgpu_nufft1d1, wgpu_nufft1d2, wgpu_nufft1d3
    public :: wgpu_nufft2d1, wgpu_nufft2d2, wgpu_nufft2d3
    public :: wgpu_nufft3d1, wgpu_nufft3d2, wgpu_nufft3d3
    public :: wgpu_nufftf_makeplan, wgpu_nufftf_setpts, wgpu_nufftf_setpts_nd
    public :: wgpu_nufftf_execute, wgpu_nufftf_destroy
    public :: wgpu_nufftf_plan_backend, wgpu_nufftf_plan_precision
    public :: wgpu_nufftf1d1, wgpu_nufftf1d2, wgpu_nufftf1d3
    public :: wgpu_nufftf2d1, wgpu_nufftf2d2, wgpu_nufftf2d3
    public :: wgpu_nufftf3d1, wgpu_nufftf3d2, wgpu_nufftf3d3

    interface
        function c_strlen(text) bind(c, name="strlen") result(length)
            import :: c_ptr, c_size_t
            type(c_ptr), value :: text
            integer(c_size_t) :: length
        end function c_strlen

        function c_last_error() bind(c, name="wgpu_nufft_last_error") result(text)
            import :: c_ptr
            type(c_ptr) :: text
        end function c_last_error

        function c_version() bind(c, name="wgpu_nufft_version") result(text)
            import :: c_ptr
            type(c_ptr) :: text
        end function c_version

        function c_gpu_name() bind(c, name="wgpu_nufft_gpu_name") result(text)
            import :: c_ptr
            type(c_ptr) :: text
        end function c_gpu_name

        subroutine c_shutdown() bind(c, name="wgpu_nufft_shutdown")
        end subroutine c_shutdown

        ! Both families share these signatures; the pointers carry the type.
        function c_makeplan(kind, dim, n_modes, isign, ntrans, eps, plan, opts) &
                bind(c, name="wgpu_nufft_makeplan") result(ier)
            import :: c_int32_t, c_int64_t, c_double, c_ptr, c_int
            integer(c_int32_t), value :: kind, dim, isign
            type(c_ptr), value :: n_modes
            integer(c_int64_t), value :: ntrans
            real(c_double), value :: eps
            type(c_ptr) :: plan
            type(c_ptr), value :: opts
            integer(c_int) :: ier
        end function c_makeplan

        function c_makeplanf(kind, dim, n_modes, isign, ntrans, eps, plan, opts) &
                bind(c, name="wgpu_nufftf_makeplan") result(ier)
            import :: c_int32_t, c_int64_t, c_double, c_ptr, c_int
            integer(c_int32_t), value :: kind, dim, isign
            type(c_ptr), value :: n_modes
            integer(c_int64_t), value :: ntrans
            real(c_double), value :: eps
            type(c_ptr) :: plan
            type(c_ptr), value :: opts
            integer(c_int) :: ier
        end function c_makeplanf

        function c_setpts(plan, m, x, y, z, n, s, t, u) bind(c, name="wgpu_nufft_setpts") result(ier)
            import :: c_ptr, c_int64_t, c_int
            type(c_ptr), value :: plan, x, y, z, s, t, u
            integer(c_int64_t), value :: m, n
            integer(c_int) :: ier
        end function c_setpts

        function c_setptsf(plan, m, x, y, z, n, s, t, u) bind(c, name="wgpu_nufftf_setpts") result(ier)
            import :: c_ptr, c_int64_t, c_int
            type(c_ptr), value :: plan, x, y, z, s, t, u
            integer(c_int64_t), value :: m, n
            integer(c_int) :: ier
        end function c_setptsf

        function c_setpts_nd(plan, m, points, n, targets) bind(c, name="wgpu_nufft_setpts_nd") result(ier)
            import :: c_ptr, c_int64_t, c_int
            type(c_ptr), value :: plan, points, targets
            integer(c_int64_t), value :: m, n
            integer(c_int) :: ier
        end function c_setpts_nd

        function c_setpts_ndf(plan, m, points, n, targets) bind(c, name="wgpu_nufftf_setpts_nd") result(ier)
            import :: c_ptr, c_int64_t, c_int
            type(c_ptr), value :: plan, points, targets
            integer(c_int64_t), value :: m, n
            integer(c_int) :: ier
        end function c_setpts_ndf

        function c_execute(plan, c, f) bind(c, name="wgpu_nufft_execute") result(ier)
            import :: c_ptr, c_int
            type(c_ptr), value :: plan, c, f
            integer(c_int) :: ier
        end function c_execute

        function c_executef(plan, c, f) bind(c, name="wgpu_nufftf_execute") result(ier)
            import :: c_ptr, c_int
            type(c_ptr), value :: plan, c, f
            integer(c_int) :: ier
        end function c_executef

        subroutine c_destroy(plan) bind(c, name="wgpu_nufft_destroy")
            import :: c_ptr
            type(c_ptr), value :: plan
        end subroutine c_destroy

        subroutine c_destroyf(plan) bind(c, name="wgpu_nufftf_destroy")
            import :: c_ptr
            type(c_ptr), value :: plan
        end subroutine c_destroyf

        function c_plan_backend(plan) bind(c, name="wgpu_nufft_plan_backend") result(value)
            import :: c_ptr, c_int
            type(c_ptr), value :: plan
            integer(c_int) :: value
        end function c_plan_backend

        function c_plan_precision(plan) bind(c, name="wgpu_nufft_plan_precision") result(value)
            import :: c_ptr, c_int
            type(c_ptr), value :: plan
            integer(c_int) :: value
        end function c_plan_precision
    end interface

contains

    ! ---------------------------------------------------------------------
    ! Library-wide routines

    function c_string(text) result(string)
        type(c_ptr), intent(in) :: text
        character(len=:), allocatable :: string
        character(kind=c_char), pointer :: chars(:)
        integer :: i, length
        if (.not. c_associated(text)) then
            string = ""
            return
        end if
        length = int(c_strlen(text))
        call c_f_pointer(text, chars, [length])
        allocate (character(len=length) :: string)
        do i = 1, length
            string(i:i) = chars(i)
        end do
    end function c_string

    ! The message of the last error on this thread.
    function wgpu_nufft_error_message() result(message)
        character(len=:), allocatable :: message
        message = c_string(c_last_error())
    end function wgpu_nufft_error_message

    function wgpu_nufft_version() result(version)
        character(len=:), allocatable :: version
        version = c_string(c_version())
    end function wgpu_nufft_version

    ! The GPU that plans run on, or an empty string without one.
    function wgpu_nufft_gpu_name() result(name)
        character(len=:), allocatable :: name
        name = c_string(c_gpu_name())
    end function wgpu_nufft_gpu_name

    ! Releases the shared GPU device and the plans the one-call routines keep.
    subroutine wgpu_nufft_shutdown()
        call c_shutdown()
    end subroutine wgpu_nufft_shutdown

    function opts_pointer(opts) result(pointer)
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        type(c_ptr) :: pointer
        pointer = c_null_ptr
        if (present(opts)) pointer = c_loc(opts)
    end function opts_pointer

    ! ---------------------------------------------------------------------
    ! Double-precision plans

    ! Creates a plan of type 1, 2 or 3 in dim dimensions for ntrans
    ! transforms per execution; types 1 and 2 take n_modes(1:dim).
    subroutine wgpu_nufft_makeplan(type, dim, n_modes, isign, ntrans, eps, plan, ier, opts)
        integer, intent(in) :: type, dim, isign, ntrans
        integer(c_int64_t), intent(in), target :: n_modes(*)
        real(c_double), intent(in) :: eps
        type(wgpu_nufft_plan), intent(out) :: plan
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        ier = c_makeplan(int(type, c_int32_t), int(dim, c_int32_t), c_loc(n_modes(1)), &
                         int(isign, c_int32_t), int(ntrans, c_int64_t), eps, plan%handle, &
                         opts_pointer(opts))
    end subroutine wgpu_nufft_makeplan

    ! Sets M points, one coordinate array per axis (y and z only in two and
    ! three dimensions), and for type 3 the N targets s, t, u.
    subroutine wgpu_nufft_setpts(plan, M, x, y, z, N, s, t, u, ier)
        type(wgpu_nufft_plan), intent(in) :: plan
        integer(c_int64_t), intent(in) :: M
        real(c_double), intent(in), target :: x(*)
        real(c_double), intent(in), optional, target :: y(*), z(*)
        integer(c_int64_t), intent(in), optional :: N
        real(c_double), intent(in), optional, target :: s(*), t(*), u(*)
        integer, intent(out) :: ier
        type(c_ptr) :: py, pz, ps, pt, pu
        integer(c_int64_t) :: targets
        py = c_null_ptr; pz = c_null_ptr; ps = c_null_ptr; pt = c_null_ptr; pu = c_null_ptr
        if (present(y)) py = c_loc(y(1))
        if (present(z)) pz = c_loc(z(1))
        if (present(s)) ps = c_loc(s(1))
        if (present(t)) pt = c_loc(t(1))
        if (present(u)) pu = c_loc(u(1))
        targets = 0
        if (present(N)) targets = N
        ier = c_setpts(plan%handle, M, c_loc(x(1)), py, pz, targets, ps, pt, pu)
    end subroutine wgpu_nufft_setpts

    ! Sets M points of any dimension as an array points(dim, M), and for
    ! type 3 the N targets targets(dim, N).
    subroutine wgpu_nufft_setpts_nd(plan, M, points, ier, N, targets)
        type(wgpu_nufft_plan), intent(in) :: plan
        integer(c_int64_t), intent(in) :: M
        real(c_double), intent(in), target :: points(*)
        integer, intent(out) :: ier
        integer(c_int64_t), intent(in), optional :: N
        real(c_double), intent(in), optional, target :: targets(*)
        type(c_ptr) :: pointer
        integer(c_int64_t) :: count
        pointer = c_null_ptr
        if (present(targets)) pointer = c_loc(targets(1))
        count = 0
        if (present(N)) count = N
        ier = c_setpts_nd(plan%handle, M, c_loc(points(1)), count, pointer)
    end subroutine wgpu_nufft_setpts_nd

    ! Runs ntrans transforms on the points last set: type 1 reads c and
    ! writes f, type 2 reads f and writes c, type 3 reads c and writes f.
    subroutine wgpu_nufft_execute(plan, c, f, ier)
        type(wgpu_nufft_plan), intent(in) :: plan
        complex(c_double_complex), intent(inout), target :: c(*), f(*)
        integer, intent(out) :: ier
        ier = c_execute(plan%handle, c_loc(c(1)), c_loc(f(1)))
    end subroutine wgpu_nufft_execute

    subroutine wgpu_nufft_destroy(plan)
        type(wgpu_nufft_plan), intent(inout) :: plan
        call c_destroy(plan%handle)
        plan%handle = c_null_ptr
    end subroutine wgpu_nufft_destroy

    ! WGPU_NUFFT_BACKEND_GPU or _CPU, the backend the plan runs on.
    integer function wgpu_nufft_plan_backend(plan)
        type(wgpu_nufft_plan), intent(in) :: plan
        wgpu_nufft_plan_backend = c_plan_backend(plan%handle)
    end function wgpu_nufft_plan_backend

    ! WGPU_NUFFT_PRECISION_F64, _DF64 or _F32, the plan's arithmetic.
    integer function wgpu_nufft_plan_precision(plan)
        type(wgpu_nufft_plan), intent(in) :: plan
        wgpu_nufft_plan_precision = c_plan_precision(plan%handle)
    end function wgpu_nufft_plan_precision

    ! ---------------------------------------------------------------------
    ! Single-precision plans

    subroutine wgpu_nufftf_makeplan(type, dim, n_modes, isign, ntrans, eps, plan, ier, opts)
        integer, intent(in) :: type, dim, isign, ntrans
        integer(c_int64_t), intent(in), target :: n_modes(*)
        real(c_double), intent(in) :: eps
        type(wgpu_nufftf_plan), intent(out) :: plan
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        ier = c_makeplanf(int(type, c_int32_t), int(dim, c_int32_t), c_loc(n_modes(1)), &
                          int(isign, c_int32_t), int(ntrans, c_int64_t), eps, plan%handle, &
                          opts_pointer(opts))
    end subroutine wgpu_nufftf_makeplan

    subroutine wgpu_nufftf_setpts(plan, M, x, y, z, N, s, t, u, ier)
        type(wgpu_nufftf_plan), intent(in) :: plan
        integer(c_int64_t), intent(in) :: M
        real(c_float), intent(in), target :: x(*)
        real(c_float), intent(in), optional, target :: y(*), z(*)
        integer(c_int64_t), intent(in), optional :: N
        real(c_float), intent(in), optional, target :: s(*), t(*), u(*)
        integer, intent(out) :: ier
        type(c_ptr) :: py, pz, ps, pt, pu
        integer(c_int64_t) :: targets
        py = c_null_ptr; pz = c_null_ptr; ps = c_null_ptr; pt = c_null_ptr; pu = c_null_ptr
        if (present(y)) py = c_loc(y(1))
        if (present(z)) pz = c_loc(z(1))
        if (present(s)) ps = c_loc(s(1))
        if (present(t)) pt = c_loc(t(1))
        if (present(u)) pu = c_loc(u(1))
        targets = 0
        if (present(N)) targets = N
        ier = c_setptsf(plan%handle, M, c_loc(x(1)), py, pz, targets, ps, pt, pu)
    end subroutine wgpu_nufftf_setpts

    subroutine wgpu_nufftf_setpts_nd(plan, M, points, ier, N, targets)
        type(wgpu_nufftf_plan), intent(in) :: plan
        integer(c_int64_t), intent(in) :: M
        real(c_float), intent(in), target :: points(*)
        integer, intent(out) :: ier
        integer(c_int64_t), intent(in), optional :: N
        real(c_float), intent(in), optional, target :: targets(*)
        type(c_ptr) :: pointer
        integer(c_int64_t) :: count
        pointer = c_null_ptr
        if (present(targets)) pointer = c_loc(targets(1))
        count = 0
        if (present(N)) count = N
        ier = c_setpts_ndf(plan%handle, M, c_loc(points(1)), count, pointer)
    end subroutine wgpu_nufftf_setpts_nd

    subroutine wgpu_nufftf_execute(plan, c, f, ier)
        type(wgpu_nufftf_plan), intent(in) :: plan
        complex(c_float_complex), intent(inout), target :: c(*), f(*)
        integer, intent(out) :: ier
        ier = c_executef(plan%handle, c_loc(c(1)), c_loc(f(1)))
    end subroutine wgpu_nufftf_execute

    subroutine wgpu_nufftf_destroy(plan)
        type(wgpu_nufftf_plan), intent(inout) :: plan
        call c_destroyf(plan%handle)
        plan%handle = c_null_ptr
    end subroutine wgpu_nufftf_destroy

    integer function wgpu_nufftf_plan_backend(plan)
        type(wgpu_nufftf_plan), intent(in) :: plan
        wgpu_nufftf_plan_backend = c_plan_backend(plan%handle)
    end function wgpu_nufftf_plan_backend

    integer function wgpu_nufftf_plan_precision(plan)
        type(wgpu_nufftf_plan), intent(in) :: plan
        wgpu_nufftf_plan_precision = c_plan_precision(plan%handle)
    end function wgpu_nufftf_plan_precision

    ! ---------------------------------------------------------------------
    ! One-call transforms, double precision. Plans are kept for reuse, so
    ! repeated calls with the same sizes and options skip plan creation.

    subroutine wgpu_nufft1d1(M, x, c, isign, eps, ms, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, ms
        real(c_double), intent(in), target :: x(*)
        complex(c_double_complex), intent(in), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_double_complex), intent(out), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, c, isign, eps, ms, f, opts) bind(c, name="wgpu_nufft1d1") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, ms
                type(c_ptr), value :: x, c, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(c(1)), int(isign, c_int32_t), eps, ms, c_loc(f(1)), opts_pointer(opts))
    end subroutine wgpu_nufft1d1

    subroutine wgpu_nufft1d2(M, x, c, isign, eps, ms, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, ms
        real(c_double), intent(in), target :: x(*)
        complex(c_double_complex), intent(out), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_double_complex), intent(in), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, c, isign, eps, ms, f, opts) bind(c, name="wgpu_nufft1d2") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, ms
                type(c_ptr), value :: x, c, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(c(1)), int(isign, c_int32_t), eps, ms, c_loc(f(1)), opts_pointer(opts))
    end subroutine wgpu_nufft1d2

    subroutine wgpu_nufft1d3(M, x, c, isign, eps, N, s, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, N
        real(c_double), intent(in), target :: x(*), s(*)
        complex(c_double_complex), intent(in), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_double_complex), intent(out), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, c, isign, eps, n, s, f, opts) bind(c, name="wgpu_nufft1d3") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, n
                type(c_ptr), value :: x, c, s, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(c(1)), int(isign, c_int32_t), eps, N, c_loc(s(1)), c_loc(f(1)), opts_pointer(opts))
    end subroutine wgpu_nufft1d3

    subroutine wgpu_nufft2d1(M, x, y, c, isign, eps, ms, mt, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, ms, mt
        real(c_double), intent(in), target :: x(*), y(*)
        complex(c_double_complex), intent(in), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_double_complex), intent(out), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, y, c, isign, eps, ms, mt, f, opts) bind(c, name="wgpu_nufft2d1") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, ms, mt
                type(c_ptr), value :: x, y, c, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(y(1)), c_loc(c(1)), int(isign, c_int32_t), eps, ms, mt, c_loc(f(1)), &
                  opts_pointer(opts))
    end subroutine wgpu_nufft2d1

    subroutine wgpu_nufft2d2(M, x, y, c, isign, eps, ms, mt, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, ms, mt
        real(c_double), intent(in), target :: x(*), y(*)
        complex(c_double_complex), intent(out), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_double_complex), intent(in), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, y, c, isign, eps, ms, mt, f, opts) bind(c, name="wgpu_nufft2d2") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, ms, mt
                type(c_ptr), value :: x, y, c, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(y(1)), c_loc(c(1)), int(isign, c_int32_t), eps, ms, mt, c_loc(f(1)), &
                  opts_pointer(opts))
    end subroutine wgpu_nufft2d2

    subroutine wgpu_nufft2d3(M, x, y, c, isign, eps, N, s, t, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, N
        real(c_double), intent(in), target :: x(*), y(*), s(*), t(*)
        complex(c_double_complex), intent(in), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_double_complex), intent(out), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, y, c, isign, eps, n, s, t, f, opts) bind(c, name="wgpu_nufft2d3") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, n
                type(c_ptr), value :: x, y, c, s, t, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(y(1)), c_loc(c(1)), int(isign, c_int32_t), eps, N, c_loc(s(1)), c_loc(t(1)), &
                  c_loc(f(1)), opts_pointer(opts))
    end subroutine wgpu_nufft2d3

    subroutine wgpu_nufft3d1(M, x, y, z, c, isign, eps, ms, mt, mu, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, ms, mt, mu
        real(c_double), intent(in), target :: x(*), y(*), z(*)
        complex(c_double_complex), intent(in), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_double_complex), intent(out), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, y, z, c, isign, eps, ms, mt, mu, f, opts) bind(c, name="wgpu_nufft3d1") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, ms, mt, mu
                type(c_ptr), value :: x, y, z, c, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(y(1)), c_loc(z(1)), c_loc(c(1)), int(isign, c_int32_t), eps, ms, mt, mu, &
                  c_loc(f(1)), opts_pointer(opts))
    end subroutine wgpu_nufft3d1

    subroutine wgpu_nufft3d2(M, x, y, z, c, isign, eps, ms, mt, mu, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, ms, mt, mu
        real(c_double), intent(in), target :: x(*), y(*), z(*)
        complex(c_double_complex), intent(out), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_double_complex), intent(in), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, y, z, c, isign, eps, ms, mt, mu, f, opts) bind(c, name="wgpu_nufft3d2") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, ms, mt, mu
                type(c_ptr), value :: x, y, z, c, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(y(1)), c_loc(z(1)), c_loc(c(1)), int(isign, c_int32_t), eps, ms, mt, mu, &
                  c_loc(f(1)), opts_pointer(opts))
    end subroutine wgpu_nufft3d2

    subroutine wgpu_nufft3d3(M, x, y, z, c, isign, eps, N, s, t, u, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, N
        real(c_double), intent(in), target :: x(*), y(*), z(*), s(*), t(*), u(*)
        complex(c_double_complex), intent(in), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_double_complex), intent(out), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, y, z, c, isign, eps, n, s, t, u, f, opts) bind(c, name="wgpu_nufft3d3") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, n
                type(c_ptr), value :: x, y, z, c, s, t, u, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(y(1)), c_loc(z(1)), c_loc(c(1)), int(isign, c_int32_t), eps, N, c_loc(s(1)), &
                  c_loc(t(1)), c_loc(u(1)), c_loc(f(1)), opts_pointer(opts))
    end subroutine wgpu_nufft3d3

    ! ---------------------------------------------------------------------
    ! One-call transforms, single precision

    subroutine wgpu_nufftf1d1(M, x, c, isign, eps, ms, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, ms
        real(c_float), intent(in), target :: x(*)
        complex(c_float_complex), intent(in), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_float_complex), intent(out), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, c, isign, eps, ms, f, opts) bind(c, name="wgpu_nufftf1d1") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, ms
                type(c_ptr), value :: x, c, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(c(1)), int(isign, c_int32_t), eps, ms, c_loc(f(1)), opts_pointer(opts))
    end subroutine wgpu_nufftf1d1

    subroutine wgpu_nufftf1d2(M, x, c, isign, eps, ms, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, ms
        real(c_float), intent(in), target :: x(*)
        complex(c_float_complex), intent(out), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_float_complex), intent(in), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, c, isign, eps, ms, f, opts) bind(c, name="wgpu_nufftf1d2") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, ms
                type(c_ptr), value :: x, c, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(c(1)), int(isign, c_int32_t), eps, ms, c_loc(f(1)), opts_pointer(opts))
    end subroutine wgpu_nufftf1d2

    subroutine wgpu_nufftf1d3(M, x, c, isign, eps, N, s, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, N
        real(c_float), intent(in), target :: x(*), s(*)
        complex(c_float_complex), intent(in), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_float_complex), intent(out), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, c, isign, eps, n, s, f, opts) bind(c, name="wgpu_nufftf1d3") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, n
                type(c_ptr), value :: x, c, s, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(c(1)), int(isign, c_int32_t), eps, N, c_loc(s(1)), c_loc(f(1)), opts_pointer(opts))
    end subroutine wgpu_nufftf1d3

    subroutine wgpu_nufftf2d1(M, x, y, c, isign, eps, ms, mt, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, ms, mt
        real(c_float), intent(in), target :: x(*), y(*)
        complex(c_float_complex), intent(in), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_float_complex), intent(out), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, y, c, isign, eps, ms, mt, f, opts) bind(c, name="wgpu_nufftf2d1") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, ms, mt
                type(c_ptr), value :: x, y, c, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(y(1)), c_loc(c(1)), int(isign, c_int32_t), eps, ms, mt, c_loc(f(1)), &
                  opts_pointer(opts))
    end subroutine wgpu_nufftf2d1

    subroutine wgpu_nufftf2d2(M, x, y, c, isign, eps, ms, mt, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, ms, mt
        real(c_float), intent(in), target :: x(*), y(*)
        complex(c_float_complex), intent(out), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_float_complex), intent(in), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, y, c, isign, eps, ms, mt, f, opts) bind(c, name="wgpu_nufftf2d2") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, ms, mt
                type(c_ptr), value :: x, y, c, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(y(1)), c_loc(c(1)), int(isign, c_int32_t), eps, ms, mt, c_loc(f(1)), &
                  opts_pointer(opts))
    end subroutine wgpu_nufftf2d2

    subroutine wgpu_nufftf2d3(M, x, y, c, isign, eps, N, s, t, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, N
        real(c_float), intent(in), target :: x(*), y(*), s(*), t(*)
        complex(c_float_complex), intent(in), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_float_complex), intent(out), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, y, c, isign, eps, n, s, t, f, opts) bind(c, name="wgpu_nufftf2d3") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, n
                type(c_ptr), value :: x, y, c, s, t, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(y(1)), c_loc(c(1)), int(isign, c_int32_t), eps, N, c_loc(s(1)), c_loc(t(1)), &
                  c_loc(f(1)), opts_pointer(opts))
    end subroutine wgpu_nufftf2d3

    subroutine wgpu_nufftf3d1(M, x, y, z, c, isign, eps, ms, mt, mu, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, ms, mt, mu
        real(c_float), intent(in), target :: x(*), y(*), z(*)
        complex(c_float_complex), intent(in), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_float_complex), intent(out), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, y, z, c, isign, eps, ms, mt, mu, f, opts) bind(c, name="wgpu_nufftf3d1") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, ms, mt, mu
                type(c_ptr), value :: x, y, z, c, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(y(1)), c_loc(z(1)), c_loc(c(1)), int(isign, c_int32_t), eps, ms, mt, mu, &
                  c_loc(f(1)), opts_pointer(opts))
    end subroutine wgpu_nufftf3d1

    subroutine wgpu_nufftf3d2(M, x, y, z, c, isign, eps, ms, mt, mu, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, ms, mt, mu
        real(c_float), intent(in), target :: x(*), y(*), z(*)
        complex(c_float_complex), intent(out), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_float_complex), intent(in), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, y, z, c, isign, eps, ms, mt, mu, f, opts) bind(c, name="wgpu_nufftf3d2") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, ms, mt, mu
                type(c_ptr), value :: x, y, z, c, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(y(1)), c_loc(z(1)), c_loc(c(1)), int(isign, c_int32_t), eps, ms, mt, mu, &
                  c_loc(f(1)), opts_pointer(opts))
    end subroutine wgpu_nufftf3d2

    subroutine wgpu_nufftf3d3(M, x, y, z, c, isign, eps, N, s, t, u, f, ier, opts)
        integer(c_int64_t), intent(in) :: M, N
        real(c_float), intent(in), target :: x(*), y(*), z(*), s(*), t(*), u(*)
        complex(c_float_complex), intent(in), target :: c(*)
        integer, intent(in) :: isign
        real(c_double), intent(in) :: eps
        complex(c_float_complex), intent(out), target :: f(*)
        integer, intent(out) :: ier
        type(wgpu_nufft_opts), intent(in), optional, target :: opts
        interface
            function api(m, x, y, z, c, isign, eps, n, s, t, u, f, opts) bind(c, name="wgpu_nufftf3d3") result(ier)
                import :: c_int64_t, c_ptr, c_int32_t, c_double, c_int
                integer(c_int64_t), value :: m, n
                type(c_ptr), value :: x, y, z, c, s, t, u, f, opts
                integer(c_int32_t), value :: isign
                real(c_double), value :: eps
                integer(c_int) :: ier
            end function api
        end interface
        ier = api(M, c_loc(x(1)), c_loc(y(1)), c_loc(z(1)), c_loc(c(1)), int(isign, c_int32_t), eps, N, c_loc(s(1)), &
                  c_loc(t(1)), c_loc(u(1)), c_loc(f(1)), opts_pointer(opts))
    end subroutine wgpu_nufftf3d3

end module wgpu_nufft
