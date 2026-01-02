#include <stdint.h>

void ref_codegen_add_f32(const float* a, const float* b, float* out, int64_t numel) {
    for (int64_t i = 0; i < numel; ++i) {
        out[i] = a[i] + b[i];
    }
}
