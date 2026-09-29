"""Static safety analysis for submitted snippets (pure, synchronous).

The sandbox kernel enforces hard resource limits; this checker adds a cheap
first line of defense: it parses Python source with :mod:`ast` and scans bash
source with regexes, emitting :class:`~sandbox_service.models.SafetyFinding`
items. Nothing here blocks the event loop and nothing touches the filesystem.

Example:
    >>> report = SafetyChecker().inspect("python", "import os; os.system('rm -rf /')")
    >>> [f.rule for f in report.blocking]
    ['dangerous-call:os.system']
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from sandbox_service.models import Language, SafetyFinding

# Builtins whose use is flagged as blocking findings.
_DANGEROUS_CALLS: dict[str, str] = {
    "eval": "dynamic code execution via eval()",
    "exec": "dynamic code execution via exec()",
    "compile": "dynamic code compilation via compile()",
    "__import__": "dynamic imports bypass static analysis",
}

# Attribute chains like os.system / shutil.rmtree are blocking.
_DANGEROUS_ATTR_CALLS: set[str] = {
    "os.system",
    "os.popen",
    "os.execv",
    "os.spawnl",
    "os.fork",
    "shutil.rmtree",
    "ctypes.pythonapi",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.run",
    "subprocess.check_output",
}

# Imports that indicate escape attempts (warned, not blocked — many are needed
# by legitimate agent code inside an already-jailed workspace).
_WATCHED_IMPORTS: set[str] = {"socket", "ctypes", "multiprocessing", "threading"}

# While True without break — infinite-loop smell (warn only; CPU limit catches it).
_INFINITE_LOOP_RE = re.compile(r"^\s*while\s+True\s*:")

def _finds_rm_rf(line: str) -> bool:
    """Detect ``rm`` invoked with both recursive and force flags (any order).

    Args:
        line: One line of bash source.

    Returns:
        True when the line contains an ``rm ... -r... -f...`` style invocation.
    """
    tokens = line.split()
    if "rm" not in [t.rsplit("/", 1)[-1] for t in tokens[:1]] and not any(
        t == "rm" or t.endswith("/rm") for t in tokens
    ):
        return False
    has_r = has_f = False
    for token in tokens:
        if token.startswith("-") and not token.startswith("--"):
            flags = set(token.lstrip("-"))
            has_r |= "r" in flags or "R" in flags
            has_f |= "f" in flags
        elif token.startswith("--"):
            has_r |= token == "--recursive"
            has_f |= token == "--force"
    return has_r and has_f


_BASH_BLOCKING_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("bash:curl-pipe-shell", "piping remote content into a shell", re.compile(r"(curl|wget)[^|]*\|\s*(ba)?sh")),
    ("bash:fork-bomb", "classic fork bomb", re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;:")),
    ("bash:sudo", "privilege escalation attempt", re.compile(r"\bsudo\b")),
    ("bash:dd-of-device", "raw write to a block device", re.compile(r"\bdd\b.*\bof=/dev/")),
]


@dataclass(frozen=True)
class SafetyReport:
    """Aggregate result of analyzing one submission.

    Attributes:
        findings: All findings, ordered by appearance.
        parse_error: Set when the source could not be parsed at all.
    """

    findings: tuple[SafetyFinding, ...] = ()
    parse_error: str | None = None

    @property
    def blocking(self) -> list[SafetyFinding]:
        """Findings severe enough to reject the submission under enforcement.

        Returns:
            List of ``severity == "block"`` findings.
        """
        return [f for f in self.findings if f.severity == "block"]

    @property
    def allowed(self) -> bool:
        """Whether the submission passes with no blocking findings.

        Returns:
            False if there are blocking findings or a parse error.
        """
        return self.parse_error is None and not self.blocking


class _CallVisitor(ast.NodeVisitor):
    """AST visitor collecting dangerous-call findings.

    Attributes:
        findings: Findings accumulated during traversal.
        blocked_calls: Dotted call names treated as blocking for this run.
    """

    def __init__(self, blocked_calls: frozenset[str]) -> None:
        """Initialize the visitor.

        Args:
            blocked_calls: Per-run set of dotted call names to block (base
                policy plus ``SafetyChecker.extra_blocked_calls``).
        """
        self.findings: list[SafetyFinding] = []
        self.blocked_calls: frozenset[str] = blocked_calls

    @staticmethod
    def _attr_chain(node: ast.Attribute) -> str | None:
        """Render a dotted attribute chain (``a.b.c``) if purely names.

        Args:
            node: Attribute AST node.

        Returns:
            The dotted string, or ``None`` if the base isn't a plain name chain.
        """
        parts: list[str] = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
        return None

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 (ast API name)
        """Flag dangerous builtins and known-risky attribute calls.

        Args:
            node: A function-call AST node.
        """
        func = node.func
        if isinstance(func, ast.Name) and func.id in _DANGEROUS_CALLS:
            self.findings.append(
                SafetyFinding(
                    rule=f"dangerous-call:{func.id}",
                    message=_DANGEROUS_CALLS[func.id],
                    line=func.lineno,
                    severity="block",
                )
            )
        elif isinstance(func, ast.Attribute):
            chain = self._attr_chain(func)
            if chain and chain in self.blocked_calls:
                self.findings.append(
                    SafetyFinding(
                        rule=f"dangerous-call:{chain}",
                        message=f"call to {chain}() can escape the intended task scope",
                        line=func.lineno,
                        severity="block",
                    )
                )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        """Warn on watched module imports.

        Args:
            node: An ``import x`` statement node.
        """
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _WATCHED_IMPORTS:
                self.findings.append(
                    SafetyFinding(
                        rule=f"watched-import:{root}",
                        message=f"module '{root}' is monitored by policy",
                        line=node.lineno,
                        severity="warn",
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        """Warn on watched from-imports.

        Args:
            node: An ``from x import y`` statement node.
        """
        root = (node.module or "").split(".")[0]
        if root in _WATCHED_IMPORTS:
            self.findings.append(
                SafetyFinding(
                    rule=f"watched-import:{root}",
                    message=f"module '{root}' is monitored by policy",
                    line=node.lineno,
                    severity="warn",
                )
            )
        self.generic_visit(node)


@dataclass
class SafetyChecker:
    """Configurable static analyzer applied before any code executes.

    Attributes:
        extra_blocked_calls: Additional dotted call names treated as blocking.
        scan_infinite_loops: Emit a warning for ``while True:`` loops.
    """

    extra_blocked_calls: tuple[str, ...] = ()
    scan_infinite_loops: bool = True

    def inspect(self, language: str | Language, source: str) -> SafetyReport:
        """Analyze one submission and return its safety report.

        Args:
            language: ``"python"`` or ``"bash"`` (enum or plain string).
            source: Program text to analyze.

        Returns:
            A :class:`SafetyReport`; never raises for malformed input —
            parse failures become findings so callers can choose policy.

        Example:
            >>> rep = SafetyChecker().inspect(Language.PYTHON, "print(1)")
            >>> rep.allowed
            True
        """
        lang = Language(language)
        if lang is Language.PYTHON:
            return self._inspect_python(source)
        return self._inspect_bash(source)

    def _inspect_python(self, source: str) -> SafetyReport:
        """Run AST-based checks over Python source.

        Args:
            source: Python program text.

        Returns:
            Findings report, including a synthetic blocking finding on
            syntax errors (unparseable code must not reach the interpreter).
        """
        try:
            tree = ast.parse(source, filename="<submission>", mode="exec")
        except SyntaxError as exc:
            finding = SafetyFinding(
                rule="syntax-error",
                message=f"source does not parse: {exc.msg} (line {exc.lineno})",
                line=exc.lineno,
                severity="block",
            )
            return SafetyReport(findings=(finding,), parse_error=str(exc))

        blocked = frozenset(_DANGEROUS_ATTR_CALLS) | frozenset(self.extra_blocked_calls)
        visitor = _CallVisitor(blocked)
        visitor.visit(tree)

        findings = list(visitor.findings)
        if self.scan_infinite_loops:
            for lineno, line in enumerate(source.splitlines(), start=1):
                if _INFINITE_LOOP_RE.match(line):
                    findings.append(
                        SafetyFinding(
                            rule="infinite-loop",
                            message="'while True' without visible exit; relying on CPU limit",
                            line=lineno,
                            severity="warn",
                        )
                    )
        return SafetyReport(findings=tuple(findings))

    def _inspect_bash(self, source: str) -> SafetyReport:
        """Run regex heuristics over bash source.

        Args:
            source: Shell script text.

        Returns:
            Findings report for matched patterns.
        """
        findings: list[SafetyFinding] = []
        for lineno, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _finds_rm_rf(line):
                findings.append(
                    SafetyFinding(
                        rule="bash:rm-rf",
                        message="recursive force-delete (rm with -r and -f)",
                        line=lineno,
                        severity="block",
                    )
                )
            for rule_id, message, pattern in _BASH_BLOCKING_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        SafetyFinding(rule=rule_id, message=message, line=lineno, severity="block")
                    )
        return SafetyReport(findings=tuple(findings))
