#include <stdint.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

static const float weight_bias[3] = {
    -0.8566746115684509f, 1.1006041765213013f, -1.0711873769760132f
};

static const float weight_weight[12] = {
    1.5409960746765137f, -0.293428897857666f, -2.1787893772125244f, 0.5684312582015991f, -1.0845223665237427f, -1.3985954523086548f, 0.40334683656692505f, 0.8380263447761536f,
    -0.7192575931549072f, -0.40334352850914f, -0.5966353416442871f, 0.18203648924827576f
};

void node1_addmm_f32(const float input[3], const float mat1[2][4], const float mat2[4][3], float out[2][3]) {
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            float acc = 0.0f;
            for (int64_t t = 0; t < 4; ++t) {
                acc += mat1[i][t] * mat2[t][j];
            }
            out[i][j] = (1.0f) * ((float*)input)[j * 1] + (1.0f) * acc;
        }
    }
}

void ref_codegen_main_f32(const float input_0[2][4], const float input_1[3], const float input_2[4][3], float out[2][3]) {
    node1_addmm_f32(input_1, input_0, input_2, out);
}

void model_run(const float* in0, float* out0) {
    ref_codegen_main_f32(in0, weight_bias, weight_weight, out0);
}
