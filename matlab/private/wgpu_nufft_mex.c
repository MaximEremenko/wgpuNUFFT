/*
 * MEX gateway of the MATLAB interface to wgpu-nufft, over the C interface
 * of capi/. build_wgpu_nufft.m compiles it with the interleaved complex API
 * (mex -R2018a); the MATLAB functions and wgpunufft_plan call it.
 *
 *   h = wgpu_nufft_mex('makeplan', type, n_modes, isign, ntrans, eps, single, opts)
 *       wgpu_nufft_mex('setpts', h, X, S)       X: M-by-dim, S: N-by-dim (type 3)
 *   o = wgpu_nufft_mex('execute', h, in, outsize)
 *       wgpu_nufft_mex('destroy', h)
 *   [backend, precision] = wgpu_nufft_mex('planinfo', h)
 *   o = wgpu_nufft_mex('simple', type, X, in, isign, eps, modes_or_S, opts)
 *   [version, gpu] = wgpu_nufft_mex('info')
 *   a = wgpu_nufft_mex('adapters')    the GPU adapters, a struct array
 *       wgpu_nufft_mex('shutdown')    releases the device and the kept plans
 *
 * opts is a struct with the fields of wgpu_nufft_opts, as wgpunufft_options
 * makes it, or empty for the defaults. Handles are uint64 values checked
 * against the live plans, so a stale handle raises an error instead of
 * crashing MATLAB.
 */
#include <stdint.h>
#include <string.h>

#include "mex.h"
#include "wgpu_nufft.h"

#define MAX_PLANS 4096
#define MAX_DIMS 16

typedef struct {
    void *plan;
    int single;
    int type;
} entry;

static entry plans[MAX_PLANS];
static int at_exit_registered = 0;

/* Destroys every plan and the shared device when MATLAB clears the MEX file
 * or exits. */
static void release_all(void) {
    int i;
    for (i = 0; i < MAX_PLANS; i++) {
        if (plans[i].plan != NULL) {
            if (plans[i].single) {
                wgpu_nufftf_destroy((wgpu_nufftf_plan)plans[i].plan);
            } else {
                wgpu_nufft_destroy((wgpu_nufft_plan)plans[i].plan);
            }
            plans[i].plan = NULL;
        }
    }
    wgpu_nufft_shutdown();
}

static void check(int code, const char *what) {
    if (code != WGPU_NUFFT_SUCCESS) {
        mexErrMsgIdAndTxt("wgpunufft:error", "%s: %s", what, wgpu_nufft_last_error());
    }
}

static double scalar(const mxArray *array, const char *name) {
    if (!mxIsNumeric(array) || mxIsComplex(array) || mxGetNumberOfElements(array) != 1) {
        mexErrMsgIdAndTxt("wgpunufft:input", "%s must be a real scalar", name);
    }
    return mxGetScalar(array);
}

static const mxArray *option(const mxArray *opts, const char *name) {
    const mxArray *value = mxGetField(opts, 0, name);
    if (value == NULL) {
        mexErrMsgIdAndTxt("wgpunufft:input", "the options lack the field %s", name);
    }
    return value;
}

/* Copies a character-vector option into a NUL-terminated field of size
 * bytes, in UTF-8. */
static void text_option(const mxArray *opts, const char *name, char *field, size_t size) {
    const mxArray *value = option(opts, name);
    char *text;
    field[0] = '\0';
    if (mxIsEmpty(value)) {
        return;
    }
    if (!mxIsChar(value)) {
        mexErrMsgIdAndTxt("wgpunufft:input", "opts.%s must be a character vector", name);
    }
    text = mxArrayToUTF8String(value);
    if (text == NULL || strlen(text) >= size) {
        if (text != NULL) {
            mxFree(text);
        }
        mexErrMsgIdAndTxt("wgpunufft:input", "opts.%s must be shorter than %d bytes", name,
                          (int)size);
    }
    memcpy(field, text, strlen(text) + 1);
    mxFree(text);
}

static wgpu_nufft_opts options(const mxArray *array) {
    wgpu_nufft_opts opts;
    wgpu_nufft_default_opts(&opts);
    if (array == NULL || mxIsEmpty(array)) {
        return opts;
    }
    if (!mxIsStruct(array) || mxGetNumberOfElements(array) != 1) {
        mexErrMsgIdAndTxt("wgpunufft:input", "the options must be a struct");
    }
    opts.backend = (int32_t)scalar(option(array, "backend"), "opts.backend");
    opts.precision = (int32_t)scalar(option(array, "precision"), "opts.precision");
    opts.mode_order = (int32_t)scalar(option(array, "mode_order"), "opts.mode_order");
    opts.threads = (int32_t)scalar(option(array, "threads"), "opts.threads");
    opts.sigma = scalar(option(array, "sigma"), "opts.sigma");
    opts.adapter_index = (int32_t)scalar(option(array, "adapter_index"), "opts.adapter_index");
    text_option(array, "adapter_name", opts.adapter_name, sizeof opts.adapter_name);
    text_option(array, "adapter_pci_bus_id", opts.adapter_pci_bus_id,
                sizeof opts.adapter_pci_bus_id);
    return opts;
}

static int handle_index(const mxArray *array) {
    uint64_T value;
    int i;
    if (!mxIsUint64(array) || mxGetNumberOfElements(array) != 1) {
        mexErrMsgIdAndTxt("wgpunufft:handle", "not a wgpu-nufft plan handle");
    }
    value = *mxGetUint64s(array);
    for (i = 0; i < MAX_PLANS; i++) {
        if (plans[i].plan != NULL && (uint64_T)(uintptr_t)plans[i].plan == value) {
            return i;
        }
    }
    mexErrMsgIdAndTxt("wgpunufft:handle",
                      "the plan no longer exists (deleted, or the MEX file was cleared)");
    return -1;
}

/* The rows of a real M-by-dim coordinate array of the plan's class. */
static mwSize coordinate_rows(const mxArray *array, mwSize dim, int single, const char *name) {
    if (mxIsComplex(array) || (single ? !mxIsSingle(array) : !mxIsDouble(array))) {
        mexErrMsgIdAndTxt("wgpunufft:input", "%s must be a real %s array", name,
                          single ? "single" : "double");
    }
    if (mxIsEmpty(array)) {
        return 0;
    }
    if ((mwSize)mxGetN(array) != dim) {
        mexErrMsgIdAndTxt("wgpunufft:input", "%s must have %d columns", name, (int)dim);
    }
    return (mwSize)mxGetM(array);
}

/* The address of column `index` of a real array, or NULL past its columns. */
static const void *column(const mxArray *array, mwSize index, int single) {
    mwSize rows;
    if (array == NULL || mxIsEmpty(array) || index >= (mwSize)mxGetN(array)) {
        return NULL;
    }
    rows = (mwSize)mxGetM(array);
    return single ? (const void *)(mxGetSingles(array) + index * rows)
                  : (const void *)(mxGetDoubles(array) + index * rows);
}

static void *complex_data(const mxArray *array, int single, const char *name) {
    if (!mxIsComplex(array) || (single ? !mxIsSingle(array) : !mxIsDouble(array))) {
        mexErrMsgIdAndTxt("wgpunufft:input", "%s must be a complex %s array", name,
                          single ? "single" : "double");
    }
    return single ? (void *)mxGetComplexSingles(array) : (void *)mxGetComplexDoubles(array);
}

static mxArray *complex_array(const mxArray *size, int single) {
    mwSize dims[MAX_DIMS + 1];
    mwSize count = (mwSize)mxGetNumberOfElements(size), i;
    const double *values;
    if (!mxIsDouble(size) || count < 1 || count > MAX_DIMS) {
        mexErrMsgIdAndTxt("wgpunufft:input", "bad output size");
    }
    values = mxGetDoubles(size);
    for (i = 0; i < count; i++) {
        dims[i] = (mwSize)values[i];
    }
    if (count == 1) {
        dims[1] = 1;
        count = 2;
    }
    return mxCreateNumericArray(count, dims, single ? mxSINGLE_CLASS : mxDOUBLE_CLASS, mxCOMPLEX);
}

static void *complex_pointer(mxArray *array, int single) {
    return single ? (void *)mxGetComplexSingles(array) : (void *)mxGetComplexDoubles(array);
}

/* Point after point: element (axis, j) of a dim-by-M copy of an M-by-dim
 * array, for plans of more than three dimensions. */
static mxArray *point_major(const mxArray *array, mwSize dim, int single) {
    mwSize rows = mxIsEmpty(array) ? 0 : (mwSize)mxGetM(array), i, j;
    mxArray *packed =
        mxCreateNumericMatrix(dim, rows, single ? mxSINGLE_CLASS : mxDOUBLE_CLASS, mxREAL);
    for (j = 0; j < rows; j++) {
        for (i = 0; i < dim; i++) {
            if (single) {
                mxGetSingles(packed)[j * dim + i] = mxGetSingles(array)[i * rows + j];
            } else {
                mxGetDoubles(packed)[j * dim + i] = mxGetDoubles(array)[i * rows + j];
            }
        }
    }
    return packed;
}

static void make_plan(mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    int64_t modes[MAX_DIMS];
    int32_t type, dim, isign, i;
    int64_t ntrans;
    double eps;
    int single, slot;
    wgpu_nufft_opts opts;
    void *plan = NULL;
    if (nrhs != 8) {
        mexErrMsgIdAndTxt("wgpunufft:input", "makeplan takes 7 arguments");
    }
    type = (int32_t)scalar(prhs[1], "type");
    if (!mxIsDouble(prhs[2]) || mxGetNumberOfElements(prhs[2]) < 1 ||
        mxGetNumberOfElements(prhs[2]) > MAX_DIMS) {
        mexErrMsgIdAndTxt("wgpunufft:input", "n_modes must be a double vector");
    }
    dim = (int32_t)mxGetNumberOfElements(prhs[2]);
    for (i = 0; i < dim; i++) {
        modes[i] = (int64_t)mxGetDoubles(prhs[2])[i];
    }
    isign = (int32_t)scalar(prhs[3], "isign");
    ntrans = (int64_t)scalar(prhs[4], "ntrans");
    eps = scalar(prhs[5], "eps");
    single = scalar(prhs[6], "single") != 0;
    opts = options(prhs[7]);
    for (slot = 0; slot < MAX_PLANS && plans[slot].plan != NULL; slot++) {
    }
    if (slot == MAX_PLANS) {
        mexErrMsgIdAndTxt("wgpunufft:plans", "too many live plans; delete some");
    }
    if (single) {
        check(wgpu_nufftf_makeplan(type, dim, modes, isign, ntrans, eps,
                                   (wgpu_nufftf_plan *)&plan, &opts),
              "makeplan");
    } else {
        check(wgpu_nufft_makeplan(type, dim, modes, isign, ntrans, eps,
                                  (wgpu_nufft_plan *)&plan, &opts),
              "makeplan");
    }
    plans[slot].plan = plan;
    plans[slot].single = single;
    plans[slot].type = type;
    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *mxGetUint64s(plhs[0]) = (uint64_T)(uintptr_t)plan;
}

static void set_points(int nrhs, const mxArray *prhs[]) {
    const mxArray *points, *targets = NULL;
    mwSize dim, m, n = 0;
    int index, single;
    if (nrhs < 3 || nrhs > 4) {
        mexErrMsgIdAndTxt("wgpunufft:input", "setpts takes a handle, X and, for type 3, S");
    }
    index = handle_index(prhs[1]);
    single = plans[index].single;
    points = prhs[2];
    dim = mxIsEmpty(points) ? 0 : (mwSize)mxGetN(points);
    if (dim == 0) {
        mexErrMsgIdAndTxt("wgpunufft:input", "X must have one column per dimension");
    }
    m = coordinate_rows(points, dim, single, "X");
    if (nrhs == 4 && !mxIsEmpty(prhs[3])) {
        targets = prhs[3];
        n = coordinate_rows(targets, dim, single, "S");
    }
    if (dim <= 3) {
        /* Columns pass through without copies. */
        if (single) {
            check(wgpu_nufftf_setpts((wgpu_nufftf_plan)plans[index].plan, (int64_t)m,
                                     column(points, 0, 1), column(points, 1, 1),
                                     column(points, 2, 1), (int64_t)n, column(targets, 0, 1),
                                     column(targets, 1, 1), column(targets, 2, 1)),
                  "setpts");
        } else {
            check(wgpu_nufft_setpts((wgpu_nufft_plan)plans[index].plan, (int64_t)m,
                                    column(points, 0, 0), column(points, 1, 0),
                                    column(points, 2, 0), (int64_t)n, column(targets, 0, 0),
                                    column(targets, 1, 0), column(targets, 2, 0)),
                  "setpts");
        }
    } else {
        mxArray *packed = point_major(points, dim, single);
        mxArray *packed_targets = targets ? point_major(targets, dim, single) : NULL;
        int code;
        if (single) {
            code = wgpu_nufftf_setpts_nd((wgpu_nufftf_plan)plans[index].plan, (int64_t)m,
                                         mxGetSingles(packed), (int64_t)n,
                                         packed_targets ? mxGetSingles(packed_targets) : NULL);
        } else {
            code = wgpu_nufft_setpts_nd((wgpu_nufft_plan)plans[index].plan, (int64_t)m,
                                        mxGetDoubles(packed), (int64_t)n,
                                        packed_targets ? mxGetDoubles(packed_targets) : NULL);
        }
        mxDestroyArray(packed);
        if (packed_targets) {
            mxDestroyArray(packed_targets);
        }
        check(code, "setpts");
    }
}

static void execute(mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    void *input, *output;
    int index, single, code;
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("wgpunufft:input", "execute takes a handle, the input and an output size");
    }
    index = handle_index(prhs[1]);
    single = plans[index].single;
    input = complex_data(prhs[2], single, "the input");
    plhs[0] = complex_array(prhs[3], single);
    output = complex_pointer(plhs[0], single);
    /* Type 2 reads f and writes c; types 1 and 3 read c and write f. */
    if (plans[index].type == 2) {
        void *swap = input;
        input = output;
        output = swap;
    }
    if (single) {
        code = wgpu_nufftf_execute((wgpu_nufftf_plan)plans[index].plan, (float *)input,
                                   (float *)output);
    } else {
        code = wgpu_nufft_execute((wgpu_nufft_plan)plans[index].plan, (double *)input,
                                  (double *)output);
    }
    check(code, "execute");
}

static void destroy(int nrhs, const mxArray *prhs[]) {
    int index;
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("wgpunufft:input", "destroy takes a handle");
    }
    index = handle_index(prhs[1]);
    if (plans[index].single) {
        wgpu_nufftf_destroy((wgpu_nufftf_plan)plans[index].plan);
    } else {
        wgpu_nufft_destroy((wgpu_nufft_plan)plans[index].plan);
    }
    plans[index].plan = NULL;
}

static const char *backend_name(int backend) {
    return backend == WGPU_NUFFT_BACKEND_GPU ? "gpu" : backend == WGPU_NUFFT_BACKEND_CPU ? "cpu" : "?";
}

static const char *precision_name(int precision) {
    switch (precision) {
    case WGPU_NUFFT_PRECISION_F64:
        return "f64";
    case WGPU_NUFFT_PRECISION_DF64:
        return "df64";
    case WGPU_NUFFT_PRECISION_F32:
        return "f32";
    default:
        return "?";
    }
}

static void plan_info(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    int index;
    void *plan;
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("wgpunufft:input", "planinfo takes a handle");
    }
    index = handle_index(prhs[1]);
    plan = plans[index].plan;
    plhs[0] = mxCreateString(backend_name(plans[index].single
                                              ? wgpu_nufftf_plan_backend((wgpu_nufftf_plan)plan)
                                              : wgpu_nufft_plan_backend((wgpu_nufft_plan)plan)));
    if (nlhs > 1) {
        plhs[1] = mxCreateString(precision_name(
            plans[index].single ? wgpu_nufftf_plan_precision((wgpu_nufftf_plan)plan)
                                : wgpu_nufft_plan_precision((wgpu_nufft_plan)plan)));
    }
}

/* One transform through the C one-call functions, which keep their plans:
 * 'simple', type, X (M-by-dim), input, isign, eps, n_modes or S, opts. */
static void simple(mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    const mxArray *points, *input_array, *last;
    int type, single, isign, code = WGPU_NUFFT_SUCCESS;
    mwSize dim, m, n = 0, i;
    int64_t modes[3] = {1, 1, 1};
    double eps;
    mxArray *outsize;
    wgpu_nufft_opts opts;
    const void *x, *y, *z, *s, *t, *u;
    void *input, *output;
    if (nrhs != 8) {
        mexErrMsgIdAndTxt("wgpunufft:input", "simple takes 7 arguments");
    }
    type = (int)scalar(prhs[1], "type");
    points = prhs[2];
    input_array = prhs[3];
    isign = (int)scalar(prhs[4], "isign");
    eps = scalar(prhs[5], "eps");
    last = prhs[6];
    opts = options(prhs[7]);
    single = mxIsSingle(points);
    dim = mxIsEmpty(points) ? 0 : (mwSize)mxGetN(points);
    if (dim < 1 || dim > 3) {
        mexErrMsgIdAndTxt("wgpunufft:input", "the one-call transforms take 1 to 3 dimensions");
    }
    m = coordinate_rows(points, dim, single, "the points");
    input = complex_data(input_array, single, type == 2 ? "f" : "c");
    x = column(points, 0, single);
    y = column(points, 1, single);
    z = column(points, 2, single);
    if (type == 3) {
        n = coordinate_rows(last, dim, single, "the target frequencies");
        s = column(last, 0, single);
        t = column(last, 1, single);
        u = column(last, 2, single);
        outsize = mxCreateDoubleMatrix(1, 1, mxREAL);
        mxGetDoubles(outsize)[0] = (double)n;
    } else {
        if (!mxIsDouble(last) || (mwSize)mxGetNumberOfElements(last) != dim) {
            mexErrMsgIdAndTxt("wgpunufft:input", "the mode counts must be %d values", (int)dim);
        }
        for (i = 0; i < dim; i++) {
            modes[i] = (int64_t)mxGetDoubles(last)[i];
        }
        outsize = mxCreateDoubleMatrix(1, type == 1 ? dim : 1, mxREAL);
        for (i = 0; i < (type == 1 ? dim : 1); i++) {
            mxGetDoubles(outsize)[i] = type == 1 ? (double)modes[i] : (double)m;
        }
        s = t = u = NULL;
    }
    plhs[0] = complex_array(outsize, single);
    mxDestroyArray(outsize);
    output = complex_pointer(plhs[0], single);
#define CALL(prefix, real)                                                                       \
    switch (type * 10 + (int)dim) {                                                              \
    case 11: code = prefix##1d1((int64_t)m, (const real *)x, (const real *)input, isign, eps,    \
                                modes[0], (real *)output, &opts); break;                         \
    case 21: code = prefix##1d2((int64_t)m, (const real *)x, (real *)output, isign, eps,         \
                                modes[0], (const real *)input, &opts); break;                    \
    case 31: code = prefix##1d3((int64_t)m, (const real *)x, (const real *)input, isign, eps,    \
                                (int64_t)n, (const real *)s, (real *)output, &opts); break;      \
    case 12: code = prefix##2d1((int64_t)m, (const real *)x, (const real *)y,                    \
                                (const real *)input, isign, eps, modes[0], modes[1],             \
                                (real *)output, &opts); break;                                   \
    case 22: code = prefix##2d2((int64_t)m, (const real *)x, (const real *)y, (real *)output,    \
                                isign, eps, modes[0], modes[1], (const real *)input, &opts);     \
        break;                                                                                   \
    case 32: code = prefix##2d3((int64_t)m, (const real *)x, (const real *)y,                    \
                                (const real *)input, isign, eps, (int64_t)n, (const real *)s,    \
                                (const real *)t, (real *)output, &opts); break;                  \
    case 13: code = prefix##3d1((int64_t)m, (const real *)x, (const real *)y, (const real *)z,   \
                                (const real *)input, isign, eps, modes[0], modes[1], modes[2],   \
                                (real *)output, &opts); break;                                   \
    case 23: code = prefix##3d2((int64_t)m, (const real *)x, (const real *)y, (const real *)z,   \
                                (real *)output, isign, eps, modes[0], modes[1], modes[2],        \
                                (const real *)input, &opts); break;                              \
    case 33: code = prefix##3d3((int64_t)m, (const real *)x, (const real *)y, (const real *)z,   \
                                (const real *)input, isign, eps, (int64_t)n, (const real *)s,    \
                                (const real *)t, (const real *)u, (real *)output, &opts); break; \
    default: mexErrMsgIdAndTxt("wgpunufft:input", "type must be 1, 2 or 3");                    \
    }
    if (single) {
        CALL(wgpu_nufftf, float)
    } else {
        CALL(wgpu_nufft, double)
    }
#undef CALL
    check(code, "transform");
}

/* The GPU adapters as an n-by-1 struct array. */
static void list_adapters(mxArray *plhs[]) {
    static const char *fields[] = {"name", "backend", "device_type", "pci_bus_id", "is_default"};
    wgpu_nufft_adapter *adapters = NULL;
    int32_t count = 0, capacity, i;
    check(wgpu_nufft_list_adapters(NULL, 0, &count), "list adapters");
    capacity = count;
    if (capacity > 0) {
        adapters = (wgpu_nufft_adapter *)mxCalloc((mwSize)capacity, sizeof *adapters);
        check(wgpu_nufft_list_adapters(adapters, capacity, &count), "list adapters");
        if (count > capacity) {
            count = capacity;
        }
    } else {
        count = 0;
    }
    plhs[0] = mxCreateStructMatrix((mwSize)count, 1, 5, fields);
    for (i = 0; i < count; i++) {
        mxSetField(plhs[0], i, "name", mxCreateString(adapters[i].name));
        mxSetField(plhs[0], i, "backend", mxCreateString(adapters[i].backend));
        mxSetField(plhs[0], i, "device_type", mxCreateString(adapters[i].device_type));
        mxSetField(plhs[0], i, "pci_bus_id", mxCreateString(adapters[i].pci_bus_id));
        mxSetField(plhs[0], i, "is_default", mxCreateLogicalScalar(adapters[i].is_default != 0));
    }
    if (adapters != NULL) {
        mxFree(adapters);
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    char command[16];
    if (!at_exit_registered) {
        mexAtExit(release_all);
        at_exit_registered = 1;
    }
    if (nrhs < 1 || !mxIsChar(prhs[0]) || mxGetString(prhs[0], command, sizeof command) != 0) {
        mexErrMsgIdAndTxt("wgpunufft:input", "the first argument must be a command");
    }
    if (strcmp(command, "makeplan") == 0) {
        make_plan(plhs, nrhs, prhs);
    } else if (strcmp(command, "setpts") == 0) {
        set_points(nrhs, prhs);
    } else if (strcmp(command, "execute") == 0) {
        execute(plhs, nrhs, prhs);
    } else if (strcmp(command, "destroy") == 0) {
        destroy(nrhs, prhs);
    } else if (strcmp(command, "planinfo") == 0) {
        plan_info(nlhs, plhs, nrhs, prhs);
    } else if (strcmp(command, "simple") == 0) {
        simple(plhs, nrhs, prhs);
    } else if (strcmp(command, "info") == 0) {
        plhs[0] = mxCreateString(wgpu_nufft_version());
        if (nlhs > 1) {
            plhs[1] = mxCreateString(wgpu_nufft_gpu_name());
        }
    } else if (strcmp(command, "adapters") == 0) {
        list_adapters(plhs);
    } else if (strcmp(command, "shutdown") == 0) {
        /* Live plans keep the device until they are deleted. */
        wgpu_nufft_shutdown();
    } else {
        mexErrMsgIdAndTxt("wgpunufft:input", "unknown command '%s'", command);
    }
}
