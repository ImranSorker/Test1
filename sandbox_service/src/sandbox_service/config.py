"""Application configuration for the sandboxed code execution service.

All tunables live in exactly one place: the :class:`Settings` model below, read
from environment variables (prefix ``SANDBOX_``) and an optional ``.env`` file.
Nothing else in the package hardcodes a path, port, image name, or secret.

Example:
    >>> import os
    >>> os.environ["SANDBOX_DEFAULT_BACKEND"] = "fake"
    >>> Settings().default_backend
    'fake'
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BackendName = Literal["local", "docker", "http", "fake"]


class Settings(BaseSettings):
    """Immutable service settings loaded from the environment.

    Attributes:
        env_file: Optional alternate ``.env`` path (tests / multi-env setups).
    """

    model_config = SettingsConfigDict(
        env_prefix="SANDBOX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **kwargs: object) -> None:
        """Create settings; ``env_file`` kwarg overrides the default ``.env``."""
        env_file = kwargs.pop("env_file", None)
        if env_file is not None:
            super().__init__(_env_file=str(env_file), **kwargs)  # type: ignore[call-arg]
        else:
            super().__init__(**kwargs)

    # --- backend selection -------------------------------------------------
    default_backend: BackendName = "local"

    # --- workspace ---------------------------------------------------------
    workspace_root: Path | None = None

    # --- limits ------------------------------------------------------------
    default_timeout_s: float = Field(default=10.0, gt=0, le=600)
    default_memory_mb: int = Field(default=512, gt=0, le=8192)
    max_output_bytes: int = Field(default=65536, ge=128, le=4 * 1024 * 1024)
    max_timeout_s: float = Field(default=120.0, gt=0, le=600)
    max_memory_mb: int = Field(default=4096, gt=0, le=8192)
    max_cpu_seconds: int = Field(default=120, gt=0, le=600)
    max_file_size_mb: int = Field(default=64, gt=0, le=512)
    max_processes: int = Field(default=128, ge=1, le=1024)

    # --- policy ------------------------------------------------------------
    allowed_languages: list[str] = ["python", "bash"]
    enforce_safety_by_default: bool = False

    # --- local adapter -----------------------------------------------------
    python_binary: str = ""
    bash_binary: str = ""

    # --- docker adapter ----------------------------------------------------
    docker_image: str = "python:3.11-slim"
    docker_cli_path: str = "docker"
    docker_container_prefix: str = "sbsvc"

    # --- http api server ---------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8090, ge=1, le=65535)
    api_auth_token: str = ""  # empty disables auth; set via SANDBOX_API_AUTH_TOKEN

    # --- resilience (remote adapters: docker/http) --------------------------
    retry_max_attempts: int = Field(default=3, ge=1, le=10)
    retry_base_delay_s: float = Field(default=0.2, gt=0, le=60)
    retry_max_delay_s: float = Field(default=5.0, gt=0, le=120)
    circuit_failure_threshold: int = Field(default=5, ge=1, le=100)
    circuit_recovery_timeout_s: float = Field(default=30.0, gt=0, le=3600)

    # --- remote http adapter -------------------------------------------------
    remote_base_url: str = ""
    remote_api_key: str = ""
    remote_request_timeout_s: float = Field(default=120.0, gt=0, le=600)

    # --- logging -------------------------------------------------------------
    log_level: str = "INFO"

    @field_validator("allowed_languages", mode="before")
    @classmethod
    def _split_languages(cls, value: object) -> object:
        """Accept ``"python,bash"`` style env values as well as lists.

        Args:
            value: Raw setting value from the environment.

        Returns:
            A list of lower-cased language names when a string was supplied.
        """
        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        return value

    @field_validator("allowed_languages")
    @classmethod
    def _known_languages(cls, value: list[str]) -> list[str]:
        """Reject unknown language names early.

        Args:
            value: Parsed language list.

        Returns:
            The validated list.

        Raises:
            ValueError: If any entry is not a supported language.
        """
        from sandbox_service.models import Language

        known = {lang.value for lang in Language}
        unknown = [item for item in value if item not in known]
        if unknown:
            raise ValueError(f"unknown languages {unknown}; supported: {sorted(known)}")
        return value

    @model_validator(mode="after")
    def _limits_are_consistent(self) -> "Settings":
        """Ensure default limits never exceed their configured maxima.

        Raises:
            ValueError: If defaults exceed maxima or delays are inconsistent.
        """
        if self.default_timeout_s > self.max_timeout_s:
            raise ValueError("SANDBOX_DEFAULT_TIMEOUT_S must be <= SANDBOX_MAX_TIMEOUT_S")
        if self.default_memory_mb > self.max_memory_mb:
            raise ValueError("SANDBOX_DEFAULT_MEMORY_MB must be <= SANDBOX_MAX_MEMORY_MB")
        if self.retry_base_delay_s > self.retry_max_delay_s:
            raise ValueError("SANDBOX_RETRY_BASE_DELAY_S must be <= SANDBOX_RETRY_MAX_DELAY_S")
        if self.default_backend == "http" and not self.remote_base_url:
            raise ValueError("SANDBOX_REMOTE_BASE_URL is required when backend is 'http'")
        if self.max_timeout_s < self.default_timeout_s:
            raise ValueError("SANDBOX_MAX_TIMEOUT_S must be >= SANDBOX_DEFAULT_TIMEOUT_S")
        return self

    @property
    def supports_rlimits(self) -> bool:
        """Whether POSIX ``resource`` limits can be applied on this platform.

        Returns:
            True on Linux/macOS, False on Windows.
        """
        return sys.platform != "win32"

    def resolved_workspace_root(self) -> Path:
        """Return the directory where per-execution workspaces are created.

        Falls back to a ``sandbox-service`` subdirectory of the system temp
        directory when ``workspace_root`` is unset.

        Returns:
            An absolute, non-existent-yet parent path (created lazily by the
            workspace manager).

        Example:
            >>> assert Settings(workspace_root=None).resolved_workspace_root().is_absolute()
        """
        if self.workspace_root is not None:
            return Path(self.workspace_root).expanduser().resolve()
        return Path(tempfile.gettempdir()) / "sandbox-service"

    def resolved_python_binary(self) -> str:
        """Return the interpreter used to run Python snippets locally.

        Prefers ``SANDBOX_PYTHON_BINARY`` (validated via ``shutil.which``),
        otherwise the current running interpreter — which guarantees the
        sandbox host process and snippet share a known binary path.

        Returns:
            Absolute path to a Python executable.

        Raises:
            ValueError: If a configured binary cannot be found on PATH.
        """
        if self.python_binary:
            found = shutil.which(self.python_binary)
            if found is None:
                raise ValueError(f"SANDBOX_PYTHON_BINARY not found: {self.python_binary!r}")
            return found
        return sys.executable

    def resolved_bash_binary(self) -> str:
        """Return the shell used to run Bash snippets locally.

        Returns:
            Absolute path to a bash executable.

        Raises:
            ValueError: If bash is not installed.
        """
        candidate = self.bash_binary or "bash"
        found = shutil.which(candidate)
        if found is None:
            raise ValueError(f"bash interpreter not found (looked for {candidate!r})")
        return found
