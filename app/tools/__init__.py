from __future__ import annotations

from app.tools.registry import ToolRegistry, builtin_registry


def load_builtin_tools() -> ToolRegistry:
    """Import the builtin support tool modules so they register themselves."""
    import app.tools.customer  # noqa: F401
    import app.tools.support  # noqa: F401
    import app.tools.transactions  # noqa: F401

    return builtin_registry()
