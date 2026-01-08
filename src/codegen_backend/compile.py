from __future__ import annotations

import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Sequence, Tuple

from distutils import ccompiler
from distutils import sysconfig as distutils_sysconfig

from codegen_backend.dtypes import _CodegenDType
from codegen_backend.graph import _GenericLibrary

_LIBRARY_CACHE: Dict[str, _GenericLibrary] = {}


def _maybe_enable_ccache(compiler: ccompiler.CCompiler) -> None:
    if compiler.compiler_type == "msvc":
        return
    ccache = os.environ.get("CCACHE") or shutil.which("ccache")
    if not ccache:
        return
    for key in ("compiler", "compiler_so", "compiler_cxx", "compiler_so_cxx"):
        cmd = compiler.executables.get(key)
        if not cmd:
            continue
        if isinstance(cmd, str):
            cmd_list = shlex.split(cmd)
        else:
            cmd_list = list(cmd)
        compiler.set_executables(**{key: " ".join([ccache, *cmd_list])})


def compile_or_load(
    kernel_src: str,
    key: str,
    *,
    entry_name: str,
    include_dirs: Sequence[Path],
    input_shapes: Tuple[Tuple[int, ...], ...],
    input_strides: Tuple[Tuple[int, ...], ...],
    output_shapes: Tuple[Tuple[int, ...], ...],
    dtype: _CodegenDType,
) -> _GenericLibrary:
    cached = _LIBRARY_CACHE.get(key)
    if cached is not None:
        return cached

    build_dir = Path(tempfile.mkdtemp(prefix="codegen_generic_"))
    c_path = build_dir / "ref_codegen_generic.c"
    c_path.write_text(kernel_src, encoding="utf-8")

    compiler = ccompiler.new_compiler()
    distutils_sysconfig.customize_compiler(compiler)
    _maybe_enable_ccache(compiler)
    compile_args: list[str]
    if compiler.compiler_type == "msvc":
        compile_args = ["/O2"]
    else:
        compile_args = ["-O3", "-fPIC"]
    objects = compiler.compile(
        [str(c_path)],
        output_dir=str(build_dir),
        include_dirs=[str(path) for path in include_dirs],
        extra_postargs=compile_args,
    )
    lib_name = "ref_codegen_generic"
    link_args: list[str] = []
    if compiler.compiler_type == "msvc":
        link_args = ["/DLL", f"/EXPORT:{entry_name}"]
    compiler.link_shared_lib(
        objects,
        lib_name,
        output_dir=str(build_dir),
        extra_postargs=link_args,
    )
    so_path = build_dir / compiler.library_filename(lib_name, lib_type="shared")

    import ctypes

    lib = ctypes.CDLL(str(so_path))
    argtypes = [ctypes.c_void_p for _ in input_shapes]
    argtypes.extend(ctypes.c_void_p for _ in output_shapes)
    getattr(lib, entry_name).argtypes = argtypes
    getattr(lib, entry_name).restype = None

    compiled = _GenericLibrary(
        so_path=so_path,
        lib=lib,
        input_shapes=input_shapes,
        input_strides=input_strides,
        output_shapes=output_shapes,
        dtype=dtype,
    )
    _LIBRARY_CACHE[key] = compiled
    return compiled
