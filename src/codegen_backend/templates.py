from __future__ import annotations

from importlib import resources

from jinja2 import Environment, FileSystemLoader

_TEMPLATE_ENV: Environment | None = None


def _build_template_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(
            resources.files("codegen_backend") / "templates"
        ),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def get_template_env() -> Environment:
    global _TEMPLATE_ENV
    if _TEMPLATE_ENV is None:
        _TEMPLATE_ENV = _build_template_env()
    return _TEMPLATE_ENV
