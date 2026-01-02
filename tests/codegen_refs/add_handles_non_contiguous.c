#include <stdint.h>
#include <stdlib.h>

void node1_add_f32(const float* a, const float* b, float* out, int64_t numel) {
    for (int64_t i = 0; i < numel; ++i) {
        out[i] = a[i] + b[i];
    }
}

void ref_codegen_main_f32(const float* input_0, const float* input_1, float* out, int64_t numel) {
    node1_add_f32(input_0, input_1, out, numel);
}
