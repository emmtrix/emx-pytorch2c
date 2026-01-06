__all__ = ["codegen_generic_backend", "export_generic_c"]


def __getattr__(name: str):
    if name == "codegen_generic_backend":
        from .backend import codegen_generic_backend

        return codegen_generic_backend
    if name == "export_generic_c":
        from .export import export_generic_c

        return export_generic_c
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
