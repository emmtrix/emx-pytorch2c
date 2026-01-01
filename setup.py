from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "ref_backend._ref_backend",
            sources=[
                "csrc/ref_backend.c",
                "csrc/ops_binary.c",
                "csrc/ops_add.c",
                "csrc/ops_sub.c",
                "csrc/ops_mul.c",
                "csrc/ops_div.c",
                "csrc/ops_maximum.c",
                "csrc/ops_minimum.c",
                "csrc/ops_neg.c",
                "csrc/ops_exp.c",
                "csrc/ops_bmm.c",
                "csrc/ops_matmul.c",
                "csrc/ops_broadcast_in_dim.c",
                "csrc/ops_unary.c",
                "csrc/ops_utils.c",
                "csrc/ref_backend_module.c",
            ],
            include_dirs=["csrc"],
        )
    ],
)
