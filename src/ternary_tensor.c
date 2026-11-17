#include "ternary_mem.h"
#include "ternary_tensor.h"
#include <stdlib.h>
#include <string.h>

static size_t shape_numel(const size_t *dims, size_t ndims) {
    size_t p = 1;
    for (size_t i = 0; i < ndims; i++) p *= dims[i];
    return ndims ? p : 0;
}

static size_t *dup_dims(const size_t *dims, size_t ndims) {
    if (ndims == 0) return NULL;
    size_t *d = ternary_malloc64(ndims * sizeof(size_t));
    if (d) memcpy(d, dims, ndims * sizeof(size_t));
    return d;
}

TernaryTensor *ternary_tensor_alloc(const size_t *dims, size_t ndims, TernaryDType dtype) {
    if (ndims == 0) return NULL;
    TernaryTensor *t = ternary_calloc64(1, sizeof(TernaryTensor));
    if (!t) return NULL;
    t->dims = dup_dims(dims, ndims);
    if (!t->dims) { free(t); return NULL; }
    t->ndims = ndims;
    t->numel = shape_numel(dims, ndims);
    t->dtype = dtype;
    t->owns_data = 1;

    switch (dtype) {
        case TERNARY_DTYPE_FLOAT:
            t->data_f = ternary_calloc64(t->numel, sizeof(float));
            if (!t->data_f) { free(t->dims); free(t); return NULL; }
            break;
        case TERNARY_DTYPE_ID:
            t->data_i = ternary_calloc64(t->numel, sizeof(int32_t));
            if (!t->data_i) { free(t->dims); free(t); return NULL; }
            break;
        case TERNARY_DTYPE_TERNARY:
            t->data_t = ternary_vec_alloc(t->numel);
            if (!t->data_t) { free(t->dims); free(t); return NULL; }
            break;
        default:
            free(t->dims); free(t); return NULL;
    }
    return t;
}

static TernaryTensor *make_view(const size_t *dims, size_t ndims, TernaryDType dtype) {
    if (ndims == 0) return NULL;
    TernaryTensor *t = ternary_calloc64(1, sizeof(TernaryTensor));
    if (!t) return NULL;
    t->dims = dup_dims(dims, ndims);
    if (!t->dims) { free(t); return NULL; }
    t->ndims = ndims;
    t->numel = shape_numel(dims, ndims);
    t->dtype = dtype;
    t->owns_data = 0;
    return t;
}

TernaryTensor *ternary_tensor_view_float(const size_t *dims, size_t ndims, float *data) {
    TernaryTensor *t = make_view(dims, ndims, TERNARY_DTYPE_FLOAT);
    if (t) t->data_f = data;
    return t;
}
TernaryTensor *ternary_tensor_view_id(const size_t *dims, size_t ndims, int32_t *data) {
    TernaryTensor *t = make_view(dims, ndims, TERNARY_DTYPE_ID);
    if (t) t->data_i = data;
    return t;
}
TernaryTensor *ternary_tensor_view_ternary(const size_t *dims, size_t ndims, TernaryVec *data) {
    TernaryTensor *t = make_view(dims, ndims, TERNARY_DTYPE_TERNARY);
    if (t) t->data_t = data;
    return t;
}

TernaryTensor *ternary_tensor_view_u8(const size_t *dims, size_t ndims, TernaryU8 *data) {
    TernaryTensor *t = make_view(dims, ndims, TERNARY_DTYPE_UINT8);
    if (t) t->data_u8 = data;
    return t;
}

void ternary_tensor_rebind_float(TernaryTensor *t, float *data) { t->data_f = data; }
void ternary_tensor_rebind_id(TernaryTensor *t, int32_t *data) { t->data_i = data; }
void ternary_tensor_rebind_ternary(TernaryTensor *t, TernaryVec *data) { t->data_t = data; }
void ternary_tensor_rebind_u8(TernaryTensor *t, TernaryU8 *data) { t->data_u8 = data; }

void ternary_tensor_free(TernaryTensor *t) {
    if (!t) return;
    if (t->owns_data) {
        free(t->data_f);
        free(t->data_i);
        if (t->data_t) ternary_vec_free(t->data_t);
    }
    free(t->dims);
    free(t);
}
