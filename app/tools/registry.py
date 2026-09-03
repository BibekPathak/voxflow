from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_args, get_origin, get_type_hints

from app.runtime.errors import ToolError, ToolNotFoundError

logger = logging.getLogger("voxflow.tools")

_TYPE_MAP: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _param_type(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    if origin is list:
        inner = get_args(annotation)
        return {"type": "array", "items": _param_type(inner[0]) if inner else {"type": "string"}}
    if origin is dict:
        return {"type": "object"}
    return {"type": _TYPE_MAP.get(annotation, "string")}


def _schema_for(fn: Callable[..., Any], descriptions: dict[str, str] | None) -> dict[str, Any]:
    hints = get_type_hints(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    signature = inspect.signature(fn)
    descriptions = descriptions or {}
    for name, parameter in signature.parameters.items():
        if name in ("self", "cls") or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        prop: dict[str, Any] = {"type": "string"}
        if name in hints:
            prop = _param_type(hints[name])
        if name in descriptions:
            prop["description"] = descriptions[name]
        properties[name] = prop
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    timeout_s: float | None = None
    max_retries: int | None = None
    retry_on_any: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass(slots=True)
class ToolOutcome:
    tool_name: str
    ok: bool
    value: Any = None
    error: str | None = None
    attempts: int = 1
    duration_ms: float = 0.0

    @property
    def content(self) -> str:
        if self.ok:
            try:
                return json.dumps(self.value, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(self.value)
        return json.dumps({"error": self.error}, ensure_ascii=False)

    def as_message(self) -> dict[str, Any]:
        return {"ok": self.ok, "tool": self.tool_name, "content": self.content}


class ToolRegistry:
    """Registers and executes tools with schema, timeout, and retry policy.

    Every tool carries an explicit JSON schema (auto-derived from the function
    signature), a description for the LLM, an execution timeout, and a retry
    policy. Execution returns a structured :class:`ToolOutcome` so the runtime
    and LLM always see a normalized result.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        self._tools[definition.name] = definition

    def register_tool(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        arg_descriptions: dict[str, str] | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
        retry_on_any: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            doc = (inspect.getdoc(func) or "").strip().splitlines()
            self.register(
                ToolDefinition(
                    name=tool_name,
                    description=description or (doc[0] if doc else tool_name),
                    parameters=_schema_for(func, arg_descriptions),
                    fn=func,
                    timeout_s=timeout_s,
                    max_retries=max_retries,
                    retry_on_any=retry_on_any,
                )
            )
            return func

        if fn is not None:
            return decorator(fn)  # type: ignore[return-value]
        return decorator

    def get(self, name: str) -> ToolDefinition:
        definition = self._tools.get(name)
        if definition is None:
            raise ToolNotFoundError(name)
        return definition

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [definition.schema() for definition in self._tools.values()]

    def definitions(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def _parse_arguments(self, name: str, arguments: str | dict[str, Any] | None) -> dict[str, Any]:
        if arguments is None or arguments == "":
            return {}
        if isinstance(arguments, dict):
            return arguments
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except (json.JSONDecodeError, TypeError) as exc:
            raise ToolError(name, f"invalid JSON arguments: {arguments!r}") from exc

    async def execute(
        self,
        name: str,
        arguments: str | dict[str, Any] | None,
        *,
        default_timeout_s: float = 10.0,
        default_retries: int = 1,
    ) -> ToolOutcome:
        definition = self.get(name)
        try:
            args = self._parse_arguments(name, arguments)
        except ToolError as exc:
            return ToolOutcome(tool_name=name, ok=False, error=str(exc), attempts=1, duration_ms=0.0)
        timeout_s = definition.timeout_s if definition.timeout_s is not None else default_timeout_s
        retries = definition.max_retries if definition.max_retries is not None else default_retries
        started = time.monotonic()
        last_error: Exception | None = None
        attempts = 0
        for _ in range(retries + 1):
            attempts += 1
            try:
                value = await asyncio.wait_for(definition.fn(**args), timeout=timeout_s)
                return ToolOutcome(
                    tool_name=name,
                    ok=True,
                    value=value,
                    attempts=attempts,
                    duration_ms=(time.monotonic() - started) * 1000,
                )
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                last_error = ToolError(name, f"timed out after {timeout_s}s", retryable=True, timeout_s=timeout_s)
            except Exception as exc:
                last_error = exc if isinstance(exc, ToolError) else ToolError(name, str(exc))
            if not (isinstance(last_error, ToolError) and last_error.retryable) and not definition.retry_on_any:
                break
            await asyncio.sleep(min(0.05 * (2 ** (attempts - 1)), 0.5))
        return ToolOutcome(
            tool_name=name,
            ok=False,
            error=str(last_error),
            attempts=attempts,
            duration_ms=(time.monotonic() - started) * 1000,
        )


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    arg_descriptions: dict[str, str] | None = None,
    timeout_s: float | None = None,
    max_retries: int | None = None,
    retry_on_any: bool = False,
) -> Any:
    """Module-level convenience decorator that registers into the builtin registry."""
    registry = _builtin_registry()
    return registry.register_tool(
        fn,
        name=name,
        description=description,
        arg_descriptions=arg_descriptions,
        timeout_s=timeout_s,
        max_retries=max_retries,
        retry_on_any=retry_on_any,
    )


_BUILTIN: ToolRegistry | None = None


def _builtin_registry() -> ToolRegistry:
    global _BUILTIN
    if _BUILTIN is None:
        _BUILTIN = ToolRegistry()
    return _BUILTIN


def builtin_registry() -> ToolRegistry:
    """Registry that the builtin support tools register themselves into."""
    return _builtin_registry()
