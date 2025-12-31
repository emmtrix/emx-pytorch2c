#include "ops_utils.h"

#include <string.h>

void write_error(char *err_msg, size_t err_cap, const char *msg) {
    if (err_msg == NULL || err_cap == 0) {
        return;
    }
    strncpy(err_msg, msg, err_cap - 1);
    err_msg[err_cap - 1] = '\0';
}

int is_contiguous(const RefTensorView *tensor) {
    if (tensor->ndim <= 0) {
        return 1;
    }
    int64_t expected = 1;
    for (int32_t i = tensor->ndim - 1; i >= 0; --i) {
        if (tensor->strides[i] != expected) {
            return 0;
        }
        expected *= tensor->sizes[i];
    }
    return 1;
}

int64_t numel(const RefTensorView *tensor) {
    if (tensor->ndim <= 0) {
        return 1;
    }
    int64_t total = 1;
    for (int32_t i = 0; i < tensor->ndim; ++i) {
        total *= tensor->sizes[i];
    }
    return total;
}
