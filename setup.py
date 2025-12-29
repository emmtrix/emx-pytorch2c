from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "ref_backend._ref_backend",
            sources=[
                "csrc/ref_backend.c",
                "csrc/ops_add.c",
                "csrc/ref_backend_module.c",
            ],
            include_dirs=["csrc"],
        )
    ],
)
