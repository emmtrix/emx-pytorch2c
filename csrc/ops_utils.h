#ifndef REF_BACKEND_OPS_UTILS_H
#define REF_BACKEND_OPS_UTILS_H

#include "ref_backend.h"

void write_error(char *err_msg, size_t err_cap, const char *msg);
int is_contiguous(const RefTensorView *tensor);
int64_t numel(const RefTensorView *tensor);

#endif
