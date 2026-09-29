"""Workspace adapter: isolated per-execution directories on the local disk.

Implements the :class:`~sandbox_service.interfaces.WorkspaceManager` port with
aiofiles-based async file injection, strict path containment (no symlink or
``..`` escapes), and a safety check that refuses to delete anything outside the
configured workspace root.

Example:
    >>> from sandbox_service.config import Settings
    >>> mgr = LocalWorkspaceManager(Settings(workspace_root="/tmp/demo-ws"))
    >>> path = mgr.create("exe_test")
    >>> path.is_dir()
    True
    >>> mgr.cleanup(path)
    >>> path.exists()
    False
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import aiofiles

from sandbox_service.config import Settings
from sandbox_service.exceptions import WorkspaceError
from sandbox_service.models import InputFile, Language
from sandbox_service.observability import get_logger


def entrypoint_filename(language: Language) -> str:
    """Return the canonical workspace filename holding a submission's source.

    Args:
        language: Submission language.

    Returns:
        ``"__main__.py"`` for Python, ``"main.sh"`` for Bash. The name is
        written by the service layer into every job workspace and referenced
        by backends through :attr:`~sandbox_service.interfaces.ExecutionSpec.entrypoint_name`,
        so adapters never need to agree on a private convention.

    Example:
        >>> entrypoint_filename(Language.PYTHON)
        '__main__.py'
    """
    return "__main__.py" if language is Language.PYTHON else "main.sh"


class LocalWorkspaceManager:
    """Creates and tears down ephemeral workspace directories.

    Attributes:
        root: Parent directory under which every workspace lives.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the manager.

        Args:
            settings: Service settings; ``resolved_workspace_root()`` selects
                the parent directory (injectable — no hidden global state).
        """
        self.root: Path = settings.resolved_workspace_root()
        self._log = get_logger(__name__, component="workspace", root=str(self.root))

    def _ensure_within_root(self, path: Path) -> Path:
        """Resolve ``path`` and verify it stays inside the workspace root.

        Args:
            path: Candidate relative-to-root or absolute path.

        Returns:
            The resolved absolute path.

        Raises:
            WorkspaceError: If resolution escapes the root (traversal/symlink).
        """
        resolved = path.resolve()
        root_resolved = self.root.resolve()
        if os.path.commonpath([str(resolved), str(root_resolved)]) != str(root_resolved):
            raise WorkspaceError(
                f"path {resolved} escapes workspace root {root_resolved}",
                details={"path": str(resolved)},
            )
        return resolved

    def create(self, execution_id: str) -> Path:
        """Create a fresh, private workspace directory for one job.

        Uses ``mkdtemp`` so concurrent jobs never collide and the directory is
        created with 0700 permissions atomically.

        Args:
            execution_id: Unique id used as the directory prefix.

        Returns:
            Absolute path of the new workspace, guaranteed to be inside
            ``self.root``.

        Raises:
            WorkspaceError: If the root cannot be created or mkdtemp fails.
        """
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            safe_prefix = "".join(ch for ch in execution_id if ch.isalnum() or ch in "-_")[:64]
            raw_path = tempfile.mkdtemp(prefix=f"{safe_prefix or 'ws'}-", dir=self.root)
        except OSError as exc:
            raise WorkspaceError(f"failed to create workspace: {exc}") from exc
        path = Path(raw_path)
        self._log.debug("workspace_created", path=str(path))
        return path

    def cleanup(self, path: Path) -> None:
        """Delete a workspace tree.

        Args:
            path: Directory previously returned by :meth:`create`.

        Raises:
            WorkspaceError: If the path is outside the root or removal fails
                for a reason other than "already gone".
        """
        target = self._ensure_within_root(path)
        if not target.exists():
            return
        try:
            shutil.rmtree(target)
            self._log.debug("workspace_removed", path=str(target))
        except OSError as exc:
            raise WorkspaceError(f"failed to remove workspace {target}: {exc}") from exc

    async def write_files(self, workspace: Path, files: list[InputFile]) -> list[Path]:
        """Inject request files into a workspace asynchronously.

        Args:
            workspace: Target directory (inside the root).
            files: Validated input files with workspace-relative paths.

        Returns:
            Absolute paths of the files written, in input order.

        Raises:
            WorkspaceError: On unsafe paths or I/O failures.
        """
        base = self._ensure_within_root(workspace)
        written: list[Path] = []
        for item in files:
            dest = self._ensure_within_root(base / item.path)
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(dest, "w", encoding="utf-8") as handle:
                    await handle.write(item.content)
            except (OSError, UnicodeEncodeError) as exc:
                raise WorkspaceError(f"failed writing {dest}: {exc}") from exc
            written.append(dest)
        self._log.debug("workspace_files_written", count=len(written), workspace=str(base))
        return written
