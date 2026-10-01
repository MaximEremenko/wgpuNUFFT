/*
 * C interface to wgpu-nufft: nonuniform fast Fourier transforms of types 1, 2
 * and 3 on the GPU (Vulkan, DX12, Metal) or on the CPU, from host memory.
 *
 * Every function returns WGPU_NUFFT_SUCCESS (0) or an error code, and
 * wgpu_nufft_last_error() then describes the error. Functions named
 * wgpu_nufft_* take double-precision arrays and wgpu_nufftf_* single-
 * precision ones; their plans are not interchangeable.
 *
 * Arrays:
 * - Complex arrays are interleaved (re, im) pairs, as C99 `double complex`,
 *   Fortran `complex(c_double_complex)` and MATLAB complex arrays store them.
 * - Fourier modes are stored with dimension zero fastest (Fortran and MATLAB
 *   column-major order, an array f(ms, mt, mu)), in centered order: index k
 *   along an axis of n modes holds frequency k - floor(n / 2). The FFT order
 *   option stores frequencies 0, 1, ..., then the negative ones.
 * - Batches of ntrans transforms that share a point set are stored one
 *   transform after another: c(M, ntrans) and f(ms, mt, mu, ntrans).
 * - Type-1 and type-2 coordinates are radians in [-3*pi, 3*pi], periodic
 *   modulo 2*pi.
 *
 * Definitions, with no normalization and isign selecting the sign:
 *   type 1: f(k) = sum_j c(j) exp(i * isign * k . x(j))
 *   type 2: c(j) = sum_k f(k) exp(i * isign * k . x(j))
 *   type 3: f(k) = sum_j c(j) exp(i * isign * s(k) . x(j))
 *
 * GPU plans share one device per process, created on first use; set the
 * WGPU_BACKEND environment variable (vulkan, dx12, metal) to pick a backend.
 * A plan may be used from one thread at a time.
 */
#ifndef WGPU_NUFFT_H
#define WGPU_NUFFT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes. */
#define WGPU_NUFFT_SUCCESS 0
/* A null pointer, a size, a point outside its range, or a call out of order. */
#define WGPU_NUFFT_ERROR_INVALID_ARGUMENT 1
/* The library rejected the plan: tolerance, dimensions, sizes or limits. */
#define WGPU_NUFFT_ERROR_PLAN 2
/* No GPU, or a GPU without what the plan needs (such as native f64). */
#define WGPU_NUFFT_ERROR_GPU_UNAVAILABLE 3
/* The GPU reported an error while running the plan. */
#define WGPU_NUFFT_ERROR_GPU 4
/* An internal error; please report it. */
#define WGPU_NUFFT_ERROR_INTERNAL 5

/* Backends (wgpu_nufft_opts.backend and wgpu_nufft_plan_backend). */
#define WGPU_NUFFT_BACKEND_AUTO 0 /* the GPU when one is available, else the CPU */
#define WGPU_NUFFT_BACKEND_GPU 1
#define WGPU_NUFFT_BACKEND_CPU 2

/* Arithmetic (wgpu_nufft_opts.precision and wgpu_nufft_plan_precision). */
/* Double-precision calls: native f64 on GPUs that support it (Vulkan), the
 * two-float Df64 format (about 44-48 bits) on others, f64 on the CPU.
 * Single-precision calls: f32. */
#define WGPU_NUFFT_PRECISION_AUTO 0
#define WGPU_NUFFT_PRECISION_F64 1  /* native f64; GPU plans fail without it */
#define WGPU_NUFFT_PRECISION_DF64 2 /* Df64 on the GPU, f64 on the CPU */
#define WGPU_NUFFT_PRECISION_F32 3  /* f32, whatever the array type */

/* Mode orders (wgpu_nufft_opts.mode_order). */
#define WGPU_NUFFT_MODE_ORDER_CENTERED 0
#define WGPU_NUFFT_MODE_ORDER_FFT 1

typedef struct wgpu_nufft_opts {
    int32_t backend;    /* WGPU_NUFFT_BACKEND_*, default AUTO */
    int32_t precision;  /* WGPU_NUFFT_PRECISION_*, default AUTO */
    int32_t mode_order; /* WGPU_NUFFT_MODE_ORDER_*, default CENTERED */
    int32_t threads;    /* CPU threads; 0 uses every available thread */
    double sigma;       /* upsampling factor; 0 selects the default, 2 */
} wgpu_nufft_opts;

/* Fills *opts with the defaults. A null opts pointer elsewhere means them. */
void wgpu_nufft_default_opts(wgpu_nufft_opts *opts);

/* The message of the last error on this thread; valid until the next call. */
const char *wgpu_nufft_last_error(void);

/* The library version, such as "0.3.0". */
const char *wgpu_nufft_version(void);

/* The GPU that plans run on, such as "<adapter name> (Vulkan)", or an empty
 * string without one. Creates the shared device on first use. */
const char *wgpu_nufft_gpu_name(void);

/* Releases the shared GPU device and the plans the one-call functions keep
 * for reuse. Plans that are still alive keep the device until destroyed. */
void wgpu_nufft_shutdown(void);

/* ---------------------------------------------------------------------- */
/* Double precision                                                        */
/* ---------------------------------------------------------------------- */

typedef struct wgpu_nufft_plan_s *wgpu_nufft_plan;

/*
 * Creates a plan of type 1, 2 or 3 in dim dimensions for ntrans transforms
 * per execution. Types 1 and 2 take n_modes[0..dim), the modes per axis;
 * type 3 ignores n_modes, which may be null.
 */
int wgpu_nufft_makeplan(int32_t type, int32_t dim, const int64_t *n_modes, int32_t isign,
                        int64_t ntrans, double eps, wgpu_nufft_plan *plan,
                        const wgpu_nufft_opts *opts);

/*
 * Sets the M nonuniform points of up to three dimensions, one coordinate
 * array per axis (y and z may be null below 2 and 3 dimensions). Type 3 also
 * takes its N target frequencies s, t, u; types 1 and 2 ignore them. The
 * arrays are copied, and executions use them until the next call.
 */
int wgpu_nufft_setpts(wgpu_nufft_plan plan, int64_t M, const double *x, const double *y,
                      const double *z, int64_t N, const double *s, const double *t,
                      const double *u);

/*
 * Sets points of any dimension, point after point: points[j * dim + axis]
 * (a Fortran array pts(dim, M)), and for type 3 targets[k * dim + axis].
 */
int wgpu_nufft_setpts_nd(wgpu_nufft_plan plan, int64_t M, const double *points, int64_t N,
                         const double *targets);

/*
 * Runs ntrans transforms on the points last set. Type 1 reads c (M by ntrans)
 * and writes f (the modes by ntrans); type 2 reads f and writes c; type 3
 * reads c and writes f (N by ntrans).
 */
int wgpu_nufft_execute(wgpu_nufft_plan plan, double *c, double *f);

/* Destroys a plan; a null plan is ignored. */
void wgpu_nufft_destroy(wgpu_nufft_plan plan);

/* The backend (WGPU_NUFFT_BACKEND_GPU or _CPU) and the arithmetic
 * (WGPU_NUFFT_PRECISION_F64, _DF64 or _F32) a plan runs with, or -1. */
int wgpu_nufft_plan_backend(wgpu_nufft_plan plan);
int wgpu_nufft_plan_precision(wgpu_nufft_plan plan);

/*
 * One-call transforms: each creates or reuses a plan for its sizes and
 * options, sets the points and runs one transform. Plans are kept for reuse
 * until wgpu_nufft_shutdown(), so repeated calls with the same sizes skip
 * plan creation.
 */
int wgpu_nufft1d1(int64_t M, const double *x, const double *c, int32_t isign, double eps,
                  int64_t ms, double *f, const wgpu_nufft_opts *opts);
int wgpu_nufft1d2(int64_t M, const double *x, double *c, int32_t isign, double eps,
                  int64_t ms, const double *f, const wgpu_nufft_opts *opts);
int wgpu_nufft1d3(int64_t M, const double *x, const double *c, int32_t isign, double eps,
                  int64_t N, const double *s, double *f, const wgpu_nufft_opts *opts);
int wgpu_nufft2d1(int64_t M, const double *x, const double *y, const double *c, int32_t isign,
                  double eps, int64_t ms, int64_t mt, double *f, const wgpu_nufft_opts *opts);
int wgpu_nufft2d2(int64_t M, const double *x, const double *y, double *c, int32_t isign,
                  double eps, int64_t ms, int64_t mt, const double *f,
                  const wgpu_nufft_opts *opts);
int wgpu_nufft2d3(int64_t M, const double *x, const double *y, const double *c, int32_t isign,
                  double eps, int64_t N, const double *s, const double *t, double *f,
                  const wgpu_nufft_opts *opts);
int wgpu_nufft3d1(int64_t M, const double *x, const double *y, const double *z, const double *c,
                  int32_t isign, double eps, int64_t ms, int64_t mt, int64_t mu, double *f,
                  const wgpu_nufft_opts *opts);
int wgpu_nufft3d2(int64_t M, const double *x, const double *y, const double *z, double *c,
                  int32_t isign, double eps, int64_t ms, int64_t mt, int64_t mu,
                  const double *f, const wgpu_nufft_opts *opts);
int wgpu_nufft3d3(int64_t M, const double *x, const double *y, const double *z, const double *c,
                  int32_t isign, double eps, int64_t N, const double *s, const double *t,
                  const double *u, double *f, const wgpu_nufft_opts *opts);

/* ---------------------------------------------------------------------- */
/* Single precision: the same functions on float arrays                    */
/* ---------------------------------------------------------------------- */

typedef struct wgpu_nufftf_plan_s *wgpu_nufftf_plan;

int wgpu_nufftf_makeplan(int32_t type, int32_t dim, const int64_t *n_modes, int32_t isign,
                         int64_t ntrans, double eps, wgpu_nufftf_plan *plan,
                         const wgpu_nufft_opts *opts);
int wgpu_nufftf_setpts(wgpu_nufftf_plan plan, int64_t M, const float *x, const float *y,
                       const float *z, int64_t N, const float *s, const float *t,
                       const float *u);
int wgpu_nufftf_setpts_nd(wgpu_nufftf_plan plan, int64_t M, const float *points, int64_t N,
                          const float *targets);
int wgpu_nufftf_execute(wgpu_nufftf_plan plan, float *c, float *f);
void wgpu_nufftf_destroy(wgpu_nufftf_plan plan);
int wgpu_nufftf_plan_backend(wgpu_nufftf_plan plan);
int wgpu_nufftf_plan_precision(wgpu_nufftf_plan plan);

int wgpu_nufftf1d1(int64_t M, const float *x, const float *c, int32_t isign, double eps,
                   int64_t ms, float *f, const wgpu_nufft_opts *opts);
int wgpu_nufftf1d2(int64_t M, const float *x, float *c, int32_t isign, double eps, int64_t ms,
                   const float *f, const wgpu_nufft_opts *opts);
int wgpu_nufftf1d3(int64_t M, const float *x, const float *c, int32_t isign, double eps,
                   int64_t N, const float *s, float *f, const wgpu_nufft_opts *opts);
int wgpu_nufftf2d1(int64_t M, const float *x, const float *y, const float *c, int32_t isign,
                   double eps, int64_t ms, int64_t mt, float *f, const wgpu_nufft_opts *opts);
int wgpu_nufftf2d2(int64_t M, const float *x, const float *y, float *c, int32_t isign,
                   double eps, int64_t ms, int64_t mt, const float *f,
                   const wgpu_nufft_opts *opts);
int wgpu_nufftf2d3(int64_t M, const float *x, const float *y, const float *c, int32_t isign,
                   double eps, int64_t N, const float *s, const float *t, float *f,
                   const wgpu_nufft_opts *opts);
int wgpu_nufftf3d1(int64_t M, const float *x, const float *y, const float *z, const float *c,
                   int32_t isign, double eps, int64_t ms, int64_t mt, int64_t mu, float *f,
                   const wgpu_nufft_opts *opts);
int wgpu_nufftf3d2(int64_t M, const float *x, const float *y, const float *z, float *c,
                   int32_t isign, double eps, int64_t ms, int64_t mt, int64_t mu,
                   const float *f, const wgpu_nufft_opts *opts);
int wgpu_nufftf3d3(int64_t M, const float *x, const float *y, const float *z, const float *c,
                   int32_t isign, double eps, int64_t N, const float *s, const float *t,
                   const float *u, float *f, const wgpu_nufft_opts *opts);

#ifdef __cplusplus
}
#endif

#endif /* WGPU_NUFFT_H */
