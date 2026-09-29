"""Unit tests for Settings validation and resolution helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from sandbox_service.config import Settings


class TestSettingsDefaults:
    def test_defaults_are_self_consistent(self) -> None:
        s = Settings()
        assert s.default_backend == "local"
        assert s.default_timeout_s <= s.max_timeout_s
        assert s.default_memory_mb <= s.max_memory_mb
        assert s.allowed_languages == ["python", "bash"]

    def test_reads_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SANDBOX_DEFAULT_BACKEND", "fake")
        monkeypatch.setenv("SANDBOX_MAX_TIMEOUT_S", "30")
        # NoDecode keeps pydantic-settings from JSON-parsing list env vars;
        # the before-validator owns comma-separated splitting/normalizing.
        monkeypatch.setenv("SANDBOX_ALLOWED_LANGUAGES", "Python, bash ")
        s = Settings()
        assert s.default_backend == "fake"
        assert s.max_timeout_s == 30
        assert s.allowed_languages == ["python", "bash"]

    def test_reads_env_file(self, tmp_path: Path) -> None:
        env = tmp_path / "custom.env"
        env.write_text("SANDBOX_API_PORT=9999\n", encoding="utf-8")
        s = Settings(env_file=env)
        assert s.api_port == 9999


class TestSettingsValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"default_timeout_s": 200.0}, "DEFAULT_TIMEOUT_S"),
            ({"default_memory_mb": 8192, "max_memory_mb": 1024}, "MEMORY"),
            ({"retry_base_delay_s": 10.0, "retry_max_delay_s": 1.0}, "RETRY"),
            ({"default_backend": "http"}, "REMOTE_BASE_URL"),
        ],
    )
    def test_inconsistent_settings_rejected(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            Settings(**kwargs)

    def test_http_backend_ok_with_url(self) -> None:
        s = Settings(default_backend="http", remote_base_url="http://kernel:8000")
        assert s.default_backend == "http"

    def test_unknown_language_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown languages"):
            Settings(allowed_languages=["python", "cobol"])

    def test_result_cache_size_bounds(self) -> None:
        assert Settings(result_cache_size=1).result_cache_size == 1
        with pytest.raises(ValueError):
            Settings(result_cache_size=0)


class TestResolvers:
    def test_workspace_root_default_is_absolute(self) -> None:
        assert Settings(workspace_root=None).resolved_workspace_root().is_absolute()

    def test_workspace_root_expands(self, tmp_path: Path) -> None:
        resolved = Settings(workspace_root=tmp_path / "ws").resolved_workspace_root()
        assert resolved == (tmp_path / "ws").resolve()

    def test_python_binary_prefers_sys_executable(self) -> None:
        import sys

        assert Settings(python_binary="").resolved_python_binary() == sys.executable

    def test_python_binary_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            Settings(python_binary="definitely-not-a-real-binary-xyz").resolved_python_binary()

    def test_supports_rlimits_matches_platform(self) -> None:
        import sys

        assert Settings().supports_rlimits is (sys.platform != "win32")
