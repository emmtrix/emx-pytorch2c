#include <stdint.h>
#include <stdlib.h>

void ref_codegen_add_f32(const float* a, const float* b, float* out, int64_t numel) {
    for (int64_t i = 0; i < numel; ++i) {
        out[i] = a[i] + b[i];
    }
}

void ref_codegen_main_f32(const float* input_0, const float* input_1, const float* input_2, float* out, int64_t numel) {
    float* tmp_0 = (float*)malloc(numel * sizeof(float));
    ref_codegen_add_f32(input_0, input_1, tmp_0, numel);
    ref_codegen_add_f32(tmp_0, input_2, out, numel);
    free(tmp_0);
}
