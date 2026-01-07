#include <stdint.h>
#include <stdbool.h>
#include "ops_scalar_f32.h"

static const float weight_initializers_onnx_initializer_0[1][16] = {
    {
        0x1.669cb3p-3f, 0x1.27a1eap-2f, 0x1.213f30p-2f, 0x1.14a67fp-1f, 0x1.17f3f7p-1f, 0x1.40cb17p-3f, -0x1.7f934ep-3f, 0x1.06eb33p-3f,
        -0x1.4f4350p-4f, -0x1.6188dep-3f, -0x1.1d3c36p-2f, -0x1.044a5ap-1f, -0x1.258e93p-2f, 0x1.4bf567p-2f, 0x1.0b24cbp-3f, -0x1.6e188ep-3f
    }
};

static const float weight_initializers_onnx_initializer_1[16] = {
    -0x1.371d99p-2f, 0x1.709bddp-3f, -0x1.4167e1p-1f, -0x1.0bebccp-15f, 0x1.679513p-2f, -0x1.083b84p+0f, 0x0.0p+0f, -0x1.1b8891p-3f,
    0x0.0p+0f, 0x0.0p+0f, 0x0.0p+0f, 0x0.0p+0f, 0x0.0p+0f, -0x1.3245f3p-1f, -0x1.030c23p-3f, 0x0.0p+0f
};

static const float weight_initializers_onnx_initializer_2[16][16] = {
    {
        -0x1.1538d4p-2f, -0x1.3e85c5p-1f, 0x1.22840dp-6f, 0x1.56044dp-2f, -0x1.33085dp-2f, -0x1.7bd78ep-3f, -0x1.57e254p-4f, 0x1.2bc583p-4f,
        -0x1.1335bap-6f, -0x1.770053p-6f, -0x1.571e38p-5f, -0x1.163000p-9f, 0x1.1ca377p-3f, -0x1.0ab317p-1f, 0x1.53b326p-7f, -0x1.3ce0acp-2f
    },
    {
        0x1.393080p-5f, 0x1.64d46bp-2f, -0x1.428fb2p-3f, 0x1.699799p-5f, 0x1.097264p-4f, -0x1.42d72ap-3f, -0x1.13d35cp-2f, 0x1.73c952p-2f,
        -0x1.7cf82fp-2f, 0x1.6250efp-2f, 0x1.3473f6p-2f, 0x1.792ac4p-4f, 0x1.132d12p-2f, -0x1.361f48p-4f, 0x1.239c43p-4f, 0x1.7958e6p-3f
    },
    {
        0x1.5320c0p-5f, -0x1.0e40f1p+0f, 0x1.2c714dp-5f, 0x1.047033p-9f, -0x1.546ccep-2f, -0x1.3cd41fp-1f, 0x1.402729p-2f, -0x1.22c84cp-1f,
        0x1.4044c7p-2f, -0x1.49e3cfp-1f, -0x1.412a21p-3f, -0x1.457d4ap-2f, 0x1.6e6512p-3f, -0x1.181b4ep+0f, 0x1.566c47p-1f, 0x1.08f89dp-2f
    },
    {
        -0x1.700600p-10f, 0x1.12a4eap-1f, 0x1.18d1abp-3f, -0x1.005c00p-5f, -0x1.031c9dp-2f, 0x1.14322ap-2f, -0x1.5a3381p-3f, 0x1.2a15f9p-2f,
        0x1.21f33ep-2f, -0x1.13f425p-2f, -0x1.64fbbbp-2f, -0x1.257088p-5f, 0x1.5071c1p-4f, 0x1.204bcap-3f, 0x1.0aa09fp-3f, -0x1.43c1b9p-3f
    },
    {
        -0x1.4da03ep-2f, 0x1.34c86ap-3f, 0x1.4d0556p-2f, 0x1.125efap-2f, -0x1.767c90p-4f, 0x1.3f2b47p-2f, -0x1.3a2bb8p-2f, 0x1.6fcdadp-4f,
        0x1.1fe72ep-2f, 0x1.781d58p-2f, 0x1.443f13p-8f, -0x1.4dae2ep-2f, -0x1.50a227p-2f, 0x1.47a46bp-1f, 0x1.6239bfp-4f, -0x1.3ccf20p-2f
    },
    {
        -0x1.223b96p-2f, -0x1.38c80dp-2f, -0x1.6620b0p-4f, 0x1.4cbf74p-1f, 0x1.13f780p-6f, -0x1.0a48b7p+0f, 0x1.16e6c1p-2f, -0x1.465879p+0f,
        0x1.198717p-1f, -0x1.709cd5p-1f, -0x1.206718p-2f, 0x1.6b85a8p-5f, 0x1.3ecd3bp-3f, -0x1.498a61p-2f, 0x1.380366p-1f, -0x1.3ce457p-2f
    },
    {
        0x1.4a591bp-2f, 0x1.15bf0cp-4f, -0x1.55fca0p-3f, -0x1.764876p-3f, 0x1.7e19a8p-5f, 0x1.54140cp-4f, 0x1.563c25p-2f, 0x1.08ea5ap-3f,
        -0x1.451040p-4f, -0x1.29a0a0p-2f, 0x1.03f0cbp-2f, -0x1.29dc95p-2f, -0x1.25d328p-2f, -0x1.0a08a6p-2f, -0x1.35a454p-2f, 0x1.371f40p-6f
    },
    {
        -0x1.546b2bp-2f, -0x1.54813fp-1f, 0x1.3ce461p-2f, 0x1.723f0fp-2f, -0x1.1e9c2bp-2f, 0x1.69735bp-3f, -0x1.70ee99p-3f, -0x1.48e3e4p-3f,
        0x1.51b35ep-4f, 0x1.2567b6p-3f, -0x1.4bdae6p-2f, 0x1.4a747cp-4f, -0x1.481ce2p-2f, -0x1.634ef9p-2f, -0x1.501896p-3f, 0x1.578420p-6f
    },
    {
        -0x1.16949ap-2f, 0x1.4eaa73p-2f, -0x1.457b80p-6f, 0x1.1eb963p-2f, -0x1.136494p-2f, 0x1.7cb714p-4f, 0x1.613496p-3f, 0x1.57f3d1p-2f,
        0x1.50fc77p-2f, -0x1.285d00p-3f, -0x1.525563p-2f, 0x1.5a7c05p-2f, 0x1.05eb05p-2f, -0x1.3e8e20p-4f, 0x1.2efc49p-2f, 0x1.23bbd1p-2f
    },
    {
        0x1.22a32fp-2f, -0x1.5feaf8p-5f, 0x1.4bdc00p-6f, 0x1.6d7caep-3f, -0x1.668468p-3f, -0x1.5c7686p-2f, -0x1.39a918p-3f, -0x1.22ea74p-3f,
        0x1.50b399p-2f, 0x1.4d3f05p-2f, -0x1.7364c0p-8f, -0x1.4cd898p-4f, -0x1.19a443p-2f, 0x1.1e294fp-2f, 0x1.498a70p-5f, -0x1.2d81e8p-4f
    },
    {
        0x1.6f99dcp-4f, -0x1.478626p-3f, 0x1.46950cp-4f, -0x1.243377p-2f, 0x1.0ee6a9p-2f, -0x1.558aa9p-2f, 0x1.4599a1p-2f, 0x1.468980p-5f,
        -0x1.00a706p-3f, -0x1.0540d8p-4f, -0x1.51af68p-2f, -0x1.218c18p-2f, -0x1.2c90f4p-3f, 0x1.64190ep-3f, -0x1.22e279p-2f, 0x1.4b9117p-2f
    },
    {
        -0x1.1d1058p-2f, -0x1.2f4e1dp-2f, 0x1.350c54p-4f, -0x1.0387ecp-2f, -0x1.556071p-3f, -0x1.06699cp-2f, 0x1.05812bp-2f, -0x1.20b502p-2f,
        0x1.54dee5p-2f, -0x1.2fdab8p-5f, -0x1.3ba5c7p-2f, -0x1.333fedp-2f, -0x1.1fdc04p-2f, 0x1.3d30e3p-2f, 0x1.0cdf9ap-3f, 0x1.0c1981p-2f
    },
    {
        0x1.0c0a3fp-2f, 0x1.188d0cp-4f, 0x1.43f965p-2f, 0x1.4d409ap-3f, 0x1.4ec362p-3f, 0x1.3886d7p-2f, -0x1.1f24adp-2f, 0x1.57b000p-8f,
        -0x1.3b4af8p-2f, 0x1.43fce5p-2f, 0x1.58be06p-3f, -0x1.476ccbp-3f, 0x1.0f7284p-4f, -0x1.00f396p-2f, 0x1.36148fp-2f, 0x1.7f5436p-3f
    },
    {
        0x1.34a8bap-3f, -0x1.7072eap-1f, -0x1.112a88p-2f, 0x1.4f7d89p-4f, 0x1.6f384ep-3f, 0x1.6340e9p-3f, 0x1.204badp-2f, -0x1.1dd4fap-2f,
        0x1.1f2c47p-2f, -0x1.2733d8p-2f, -0x1.2be7c8p-4f, 0x1.3dad4cp-4f, 0x1.6dd830p-5f, -0x1.0253f1p+0f, 0x1.5debeap-2f, -0x1.236050p-4f
    },
    {
        0x1.036e00p-7f, -0x1.082aa1p+0f, 0x1.43a6b0p-5f, -0x1.4ab9dcp-2f, -0x1.35e80ep-2f, -0x1.277b4dp-5f, 0x1.19938dp-2f, 0x1.0175eap-1f,
        -0x1.0794d0p-4f, 0x1.5099bep-2f, 0x1.4ba53ep-3f, 0x1.5b3e78p-5f, -0x1.7a82d9p-5f, -0x1.08a3fbp+0f, 0x1.06db10p-2f, -0x1.575c7ap-3f
    },
    {
        -0x1.21fe9ap-2f, 0x1.377102p-3f, 0x1.3cf2c7p-2f, 0x1.2e26c8p-5f, 0x1.0b5142p-3f, 0x1.0d913dp-2f, -0x1.16352ap-2f, 0x1.0aa5a6p-3f,
        0x1.411c6ep-3f, -0x1.410f5ep-2f, 0x1.0f6e9fp-2f, -0x1.09090dp-2f, -0x1.2f2808p-4f, 0x1.755040p-5f, 0x1.1ec69fp-2f, 0x1.0ba6d1p-2f
    }
};

static const float weight_initializers_onnx_initializer_3[16] = {
    0x0.0p+0f, 0x1.239f79p-4f, 0x1.249124p-3f, -0x1.2f5076p-2f, 0x0.0p+0f, 0x1.4cc498p-3f, 0x0.0p+0f, 0x1.0e848bp-3f,
    -0x1.4796a0p-2f, 0x1.3279a0p-2f, -0x1.52f5a8p-4f, 0x0.0p+0f, -0x1.0856f7p-5f, 0x1.721db0p-2f, -0x1.485319p-2f, 0x0.0p+0f
};

static const float weight_initializers_onnx_initializer_4[16][1] = {
    {
        0x1.257da0p-2f
    },
    {
        0x1.26d0c7p+0f
    },
    {
        -0x1.218ba4p-2f
    },
    {
        0x1.1038d0p-1f
    },
    {
        0x1.774494p-2f
    },
    {
        -0x1.08a91bp-1f
    },
    {
        0x1.37f39ap-2f
    },
    {
        -0x1.50261ep-3f
    },
    {
        0x1.3b3c10p-2f
    },
    {
        -0x1.2fd25cp-1f
    },
    {
        0x1.1e7483p-4f
    },
    {
        -0x1.20111cp-2f
    },
    {
        0x1.75d693p-2f
    },
    {
        0x1.17efa6p+0f
    },
    {
        0x1.7d598dp-2f
    },
    {
        0x1.248e70p-4f
    }
};

static const float weight_initializers_onnx_initializer_5[1] = {
    -0x1.712c2bp-2f
};

void node1_matmul_f32(const float a[1][1], const float b[1][16], float out[1][16]) {
        for (int64_t i = 0; i < 1; ++i) {
            for (int64_t j = 0; j < 16; ++j) {
                float acc = 0.0f;
                for (int64_t t = 0; t < 1; ++t) {
                    acc += a[i][t] * b[t][j];
                }
                out[i][j] = acc;
            }
        }
}

void node2_add_f32(const float a[1][16], const float b[16], float out[1][16]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 16; ++i1) {
            out[i0][i1] = ref_scalar_f32_add(a[0][i1], b[i1]);
        }
    }
}

void node3_relu_f32(const float a[1][16], float out[1][16]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 16; ++i1) {
            out[i0][i1] = ref_scalar_f32_relu(a[i0][i1]);
        }
    }
}

void node4_matmul_f32(const float a[1][16], const float b[16][16], float out[1][16]) {
        for (int64_t i = 0; i < 1; ++i) {
            for (int64_t j = 0; j < 16; ++j) {
                float acc = 0.0f;
                for (int64_t t = 0; t < 16; ++t) {
                    acc += a[i][t] * b[t][j];
                }
                out[i][j] = acc;
            }
        }
}

void node5_add_f32(const float a[1][16], const float b[16], float out[1][16]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 16; ++i1) {
            out[i0][i1] = ref_scalar_f32_add(a[0][i1], b[i1]);
        }
    }
}

void node6_relu_f32(const float a[1][16], float out[1][16]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 16; ++i1) {
            out[i0][i1] = ref_scalar_f32_relu(a[i0][i1]);
        }
    }
}

void node7_matmul_f32(const float a[1][16], const float b[16][1], float out[1][1]) {
        for (int64_t i = 0; i < 1; ++i) {
            for (int64_t j = 0; j < 1; ++j) {
                float acc = 0.0f;
                for (int64_t t = 0; t < 16; ++t) {
                    acc += a[i][t] * b[t][j];
                }
                out[i][j] = acc;
            }
        }
}

void node8_add_f32(const float a[1][1], const float b[1], float out[1][1]) {
    for (int64_t i0 = 0; i0 < 1; ++i0) {
        for (int64_t i1 = 0; i1 < 1; ++i1) {
            out[i0][i1] = ref_scalar_f32_add(a[0][0], b[0]);
        }
    }
}

void ref_codegen_main_f32(const float input_0[1][1], const float input_1[1][16], const float input_2[16], const float input_3[16][16], const float input_4[16], const float input_5[16][1], const float input_6[1], float out[1][1]) {
    float tmp_0[1][16];
    float tmp_1[1][16];
    float tmp_2[1][16];
    float tmp_3[1][16];
    float tmp_4[1][16];
    float tmp_5[1][16];
    float tmp_6[1][1];
    node1_matmul_f32(input_0, input_1, tmp_0);
    node2_add_f32(tmp_0, input_2, tmp_1);
    node3_relu_f32(tmp_1, tmp_2);
    node4_matmul_f32(tmp_2, input_3, tmp_3);
    node5_add_f32(tmp_3, input_4, tmp_4);
    node6_relu_f32(tmp_4, tmp_5);
    node7_matmul_f32(tmp_5, input_5, tmp_6);
    node8_add_f32(tmp_6, input_6, out);
}

void entry(const float* in0, float* out0) {
    ref_codegen_main_f32(in0, weight_initializers_onnx_initializer_0, weight_initializers_onnx_initializer_1, weight_initializers_onnx_initializer_2, weight_initializers_onnx_initializer_3, weight_initializers_onnx_initializer_4, weight_initializers_onnx_initializer_5, out0);
}
