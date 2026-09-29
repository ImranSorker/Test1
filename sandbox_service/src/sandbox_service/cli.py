"""Command-line surface: ``sandbox run|serve|health|languages``.

Built on Typer + Rich; every command delegates to the same
:class:`~sandbox_service.service.SandboxService` used by the HTTP API, so CLI
behaviour is exactly the Python-API behaviour with a nicer presentation.

Exit codes (stable contract for agent scripts):

* ``0`` request completed (even if the *snippet* failed — inspect status)
* ``1`` snippet did not succeed (non-zero exit / timeout / kernel error)
* ``2`` request rejected (validation, unsupported language, safety policy)
* ``3`` configuration or backend-startup failure

Example:
    >>> from typer.testing import CliRunner
    >>> from sandbox_service.cli import app
    >>> result = CliRunner().invoke(app, ["--help"])
    >>> result.exit_code
    0
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.syntax import Syntax

from sandbox_service.config import Settings
from sandbox_service.exceptions import SandboxError
from sandbox_service.models import ExecutionRequest, ExecutionResult, Language, ResourceLimits
from sandbox_service.observability import configure_logging
from sandbox_service.service import SandboxService

app = typer.Typer(
    name="sandbox",
    help="Run untrusted Python/Bash snippets in an isolated, resource-limited sandbox.",
    add_completion=False,
    no_args_is_help=True,
)
_console = Console(stderr=False)
_err_console = Console(stderr=True)

# Exit-code contract (documented at module level and in --help).
EXIT_OK = 0
EXIT_SNIPPET_FAILED = 1
EXIT_REJECTED = 2
EXIT_CONFIG = 3


def _load_settings(env_file: Path | None) -> Settings:
    """Construct settings, exiting with code 3 on misconfiguration.

    Args:
        env_file: Optional alternate ``.env`` path.

    Returns:
        Validated :class:`~sandbox_service.config.Settings`.

    Raises:
        typer.Exit: Always exits non-zero when validation fails; never
            returns an invalid object.
    """
    try:
        settings = Settings(env_file=env_file) if env_file else Settings()
    except Exception as exc:  # noqa: BLE001 - pydantic-settings raises many types
        _err_console.print(f"[red]configuration error:[/red] {exc}")
        raise typer.Exit(code=EXIT_CONFIG) from exc
    configure_logging(settings.log_level)
    return settings


def _render_result(result: ExecutionResult, *, as_json: bool) -> None:
    """Print an execution result either as JSON or pretty Rich panels.

    Args:
        result: The terminal outcome to display.
        as_json: When True emit machine-readable JSON on stdout.
    """
    if as_json:
        _console.print_json(result.model_dump_json())
        return
    colors = {
        "succeeded": "green",
        "failed": "red",
        "timed_out": "yellow",
        "rejected": "magenta",
        "error": "red",
    }
    color = colors.get(result.status.value, "white")
    header = (
        f"[{color}]{result.status.value}[/{color}] "
        f"exit={result.exit_code if result.exit_code is not None else '-'} "
        f"{result.duration_ms}ms backend={result.backend} id={result.execution_id}"
    )
    _console.rule(header)
    if result.stdout:
        _console.print("[bold]stdout:[/bold]")
        _console.print(Syntax(result.stdout.rstrip("\n"), "text", background_color="default"))
    if result.stderr:
        _console.print("[bold]stderr:[/bold]")
        _console.print(Syntax(result.stderr.rstrip("\n"), "text", background_color="default"))
    if result.truncated:
        _err_console.print("[yellow]warning: output truncated to byte budget[/yellow]")
    for finding in result.findings:
        _err_console.print(
            f"[{'red' if finding.severity == 'block' else 'yellow'}]"
            f"[{finding.severity}] {finding.rule}: {finding.message}"
            f"{' (line ' + str(finding.line) + ')' if finding.line else ''}[/]"
        )


def _exit_code_for(result: ExecutionResult) -> int:
    """Map a terminal result onto the documented CLI exit code.

    Args:
        result: Completed execution outcome.

    Returns:
        ``EXIT_OK`` for successes, ``EXIT_SNIPPET_FAILED`` otherwise.
    """
    return EXIT_OK if result.ok else EXIT_SNIPPET_FAILED


async def _run_request(settings: Settings, request: ExecutionRequest, *, as_json: bool) -> int:
    """Execute one request through a freshly built service.

    Args:
        settings: Resolved configuration.
        request: Validated submission.
        as_json: Emit JSON instead of pretty output.

    Returns:
        Process exit code per the module-level contract.
    """
    try:
        async with SandboxService(settings) as svc:
            result = await svc.execute(request)
    except SandboxError as exc:
        payload: dict[str, Any] = exc.to_dict()
        if as_json:
            _console.print_json(json.dumps(payload))
        else:
            _err_console.print(f"[magenta]rejected[{payload['error']}]:[/magenta] {payload['message']}")
        return EXIT_REJECTED
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary
        _err_console.print(f"[red]fatal:[/red] {type(exc).__name__}: {exc}")
        return EXIT_CONFIG
    _render_result(result, as_json=as_json)
    return _exit_code_for(result)


@app.command()
def run(
    source: Annotated[
        str | None,
        typer.Argument(help="Program text. Use '-' to read from stdin."),
    ] = None,
    language: Annotated[
        Language,
        typer.Option("--language", "--lang", "-l", help="Interpreter to use."),
    ] = Language.PYTHON,
    file: Annotated[
        Path | None,
        typer.Option("--file", "-f", help="Read program text from this file instead."),
    ] = None,
    arg: Annotated[
        list[str],
        typer.Option("--arg", "-a", help="Argument passed to the program (repeatable)."),
    ] = [],
    stdin_text: Annotated[
        str,
        typer.Option("--stdin", help="Text fed to the program's standard input."),
    ] = "",
    timeout: Annotated[
        float | None,
        typer.Option("--timeout", "-t", help="Wall-clock limit in seconds (clamped to max)."),
    ] = None,
    memory_mb: Annotated[
        int | None,
        typer.Option("--memory-mb", "-m", help="Address-space cap in MiB (clamped to max)."),
    ] = None,
    enforce_safety: Annotated[
        bool,
        typer.Option("--enforce-safety/--no-enforce-safety", help="Reject blocking findings."),
    ] = False,
    json_out: Annotated[
        bool,
        typer.Option("--json", "-j", help="Emit machine-readable JSON on stdout."),
    ] = False,
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", help="Alternate .env file for settings."),
    ] = None,
) -> None:
    """Execute a snippet in the sandbox and print its result.

    Args:
        source: Inline program text, or ``-`` to pipe from stdin.
        language: Interpreter selection (python/bash).
        file: Load program text from ``file`` instead of ``source``.
        arg: Repeatable argv passed to the program.
        stdin_text: Bytes fed to the program's stdin.
        timeout: Wall-clock override (bounded by server config).
        memory_mb: Memory override (bounded by server config).
        enforce_safety: Turn static-analysis warnings-as-errors on/off.
        json_out: Emit the full :class:`ExecutionResult` as JSON.
        env_file: Alternate environment file for settings.

    Raises:
        typer.Exit: Always — carries the documented process exit code.
    """
    settings = _load_settings(env_file)
    if file is not None:
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as exc:
            _err_console.print(f"[red]cannot read {file}:[/red] {exc}")
            raise typer.Exit(code=EXIT_CONFIG) from exc
    elif source == "-":
        text = sys.stdin.read()
    elif source is not None:
        text = source
    else:
        _err_console.print("provide SOURCE, pass '-', or use --file")
        raise typer.Exit(code=EXIT_CONFIG)

    limits_kwargs: dict[str, float | int] = {}
    if timeout is not None:
        limits_kwargs["timeout_seconds"] = timeout
    if memory_mb is not None:
        limits_kwargs["memory_mb"] = memory_mb
    request = ExecutionRequest(
        language=language,
        source=text,
        args=list(arg),
        stdin=stdin_text,
        enforce_safety=enforce_safety,
        limits=ResourceLimits(**limits_kwargs) if limits_kwargs else None,
    )
    raise typer.Exit(code=asyncio.run(_run_request(settings, request, as_json=json_out)))


@app.command()
def health(
    json_out: Annotated[bool, typer.Option("--json", "-j", help="Emit JSON report.")] = False,
    env_file: Annotated[Path | None, typer.Option("--env-file")] = None,
) -> None:
    """Probe the configured backend and circuit-breaker state.

    Args:
        json_out: Emit the :class:`HealthReport` as JSON.
        env_file: Alternate environment file for settings.

    Raises:
        typer.Exit: 0 healthy, 3 unhealthy/config error.
    """
    settings = _load_settings(env_file)

    async def _probe() -> int:
        try:
            async with SandboxService(settings) as svc:
                report = await svc.health()
        except SandboxError as exc:
            _err_console.print(f"[red]backend error:[/red] {exc}")
            return EXIT_CONFIG
        if json_out:
            _console.print_json(report.model_dump_json())
        else:
            mark = "[green]healthy[/green]" if report.healthy else "[red]unhealthy[/red]"
            _console.print(f"{mark} backend=[bold]{report.backend}[/bold] circuit={report.circuit_state}")
            for key, value in report.details.items():
                _console.print(f"  {key}: {value}")
        return EXIT_OK if report.healthy else EXIT_CONFIG

    raise typer.Exit(code=asyncio.run(_probe()))


@app.command()
def languages(
    env_file: Annotated[Path | None, typer.Option("--env-file")] = None,
) -> None:
    """List languages enabled by the current configuration.

    Args:
        env_file: Alternate environment file for settings.

    Raises:
        typer.Exit: Always 0 after printing the allowlist.
    """
    settings = _load_settings(env_file)
    for item in sorted(settings.allowed_languages):
        _console.print(item)
    raise typer.Exit(code=EXIT_OK)


@app.command()
def serve(
    host: Annotated[str | None, typer.Option("--host", help="Bind address (default from config).")] = None,
    port: Annotated[int | None, typer.Option("--port", "-p", help="Bind port (default from config).")] = None,
    reload: Annotated[bool, typer.Option("--reload", help="Dev auto-reload.")] = False,
    env_file: Annotated[Path | None, typer.Option("--env-file")] = None,
) -> None:
    """Start the FastAPI HTTP server (uvicorn) with current settings.

    Args:
        host: Bind address override.
        port: Bind port override.
        reload: Enable uvicorn autoreload (development only).
        env_file: Alternate environment file for settings.

    Raises:
        typer.Exit: uvicorn's own exit status propagates.
    """
    settings = _load_settings(env_file)
    import uvicorn

    from sandbox_service.api import create_app

    uvicorn.run(
        create_app(settings),
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


def main() -> None:  # pragma: no cover - thin wrapper
    """Console-script entry point (``sandbox``)."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
