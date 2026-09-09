"""Repair the virtual environment before the tools try to import into it.

The failure this exists for: on Windows ``Activate.ps1`` is refused by the
default execution policy, PowerShell continues anyway, ``pip install`` lands in
the global interpreter, and VS Code then runs the scripts with the empty
``.venv`` it auto-detected. The user sees ``No module named 'numpy'`` having
followed the README exactly.

Import this first, before any third-party import, in any entry point:

    import env_bootstrap
    env_bootstrap.ensure_environment()

When everything is installed this costs one metadata lookup per requirement and
returns. When something is missing it installs ``requirements.txt`` into the
right interpreter and relaunches the original command there, so the user's next
sight is the working tool rather than a traceback.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
REQUIREMENTS_FILE = REPO_DIR / "requirements.txt"
VENV_DIR = REPO_DIR / ".venv"

# Set in the relaunched child so a still-broken environment fails with a real
# message instead of forking forever.
GUARD_ENV_VAR = "SAM_MEASURER_ENV_REPAIRED"
# Escape hatch for anyone managing the environment themselves.
SKIP_ENV_VAR = "SAM_MEASURER_SKIP_BOOTSTRAP"
# Suppresses the failure popup, which would otherwise block an unattended run.
NO_DIALOG_ENV_VAR = "SAM_MEASURER_NO_DIALOG"

_REQUIREMENT_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(?:(>=|==|~=|>)\s*([0-9][^,;\s]*))?"
)


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _same_interpreter(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _parse_requirements() -> list[tuple[str, str | None, str | None]]:
    entries: list[tuple[str, str | None, str | None]] = []
    for raw_line in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _REQUIREMENT_RE.match(line)
        if match:
            entries.append((match.group(1), match.group(2), match.group(3)))
    return entries


def _numeric_prefix(version_text: str) -> tuple[int, ...]:
    """Leading numeric release segment, so ``1.26.4rc1`` compares as ``(1, 26, 4)``."""
    parts: list[int] = []
    for chunk in re.split(r"[._-]", version_text):
        digits = re.match(r"^\d+", chunk)
        if not digits:
            break
        parts.append(int(digits.group(0)))
        if digits.group(0) != chunk:
            break
    return tuple(parts)


def missing_requirements() -> list[str]:
    """Requirement names absent from, or older than the pin in, this interpreter."""
    from importlib import metadata

    missing: list[str] = []
    for name, operator, wanted in _parse_requirements():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if operator in (">=", "==", "~=") and wanted:
            have, need = _numeric_prefix(installed), _numeric_prefix(wanted)
            if have and need and have < need:
                missing.append(name)
    return missing


def _report(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _show_dialog(title: str, message: str) -> None:
    """Best-effort popup for users who launched without a readable terminal.

    Modal, so it is skipped whenever stderr is a terminal the user can already
    read, and whenever the popup is suppressed outright.
    """
    if os.environ.get(NO_DIALOG_ENV_VAR):
        return
    try:
        if sys.stderr is not None and sys.stderr.isatty():
            return
    except Exception:
        pass
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        pass


def _abort(message: str) -> None:
    manual = (
        f"{message}\n\n"
        "Fix it by hand from the antscihub-sam-measurer folder:\n"
        f"  {_venv_python()} -m pip install -r {REQUIREMENTS_FILE}\n\n"
        "If that path does not exist, create the environment first:\n"
        f"  {sys.executable} -m venv .venv"
    )
    _report(f"\n{manual}\n")
    _show_dialog("SAM measurer: environment not ready", manual)
    raise SystemExit(1)


def _run(command: list[str], description: str) -> None:
    _report(f"[env] {description}")
    result = subprocess.run(command)
    if result.returncode != 0:
        _abort(f"{description} failed (exit code {result.returncode}).")


def _create_venv(target_python: Path) -> None:
    _run([sys.executable, "-m", "venv", str(VENV_DIR)], f"Creating {VENV_DIR}")
    if not target_python.exists():
        _abort(f"Created {VENV_DIR} but {target_python} is missing.")


def _install(target_python: Path) -> None:
    has_pip = subprocess.run(
        [str(target_python), "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not has_pip:
        _run([str(target_python), "-m", "ensurepip", "--upgrade"], "Installing pip")
    _run(
        [str(target_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        f"Installing requirements into {target_python}",
    )


def _relaunch(target_python: Path) -> None:
    child_env = dict(os.environ)
    child_env[GUARD_ENV_VAR] = "1"
    _report(f"[env] Restarting with {target_python}\n")
    result = subprocess.run([str(target_python), *sys.argv], env=child_env)
    raise SystemExit(result.returncode)


def ensure_environment() -> None:
    """Install missing requirements and relaunch, or return if nothing is wrong."""
    if os.environ.get(SKIP_ENV_VAR) or not REQUIREMENTS_FILE.exists():
        return

    missing = missing_requirements()
    if not missing:
        return

    listed = ", ".join(missing)
    if os.environ.get(GUARD_ENV_VAR):
        _abort(f"Still missing after a repair attempt: {listed}.")

    _report(f"\n[env] Missing or outdated: {listed}")
    _report(f"[env] Current interpreter: {sys.executable}")

    inside_a_virtualenv = sys.prefix != sys.base_prefix
    if inside_a_virtualenv:
        # Respect whatever environment the user is deliberately running in,
        # including conda, rather than building a second one beside it.
        target_python = Path(sys.executable)
    else:
        target_python = _venv_python()
        if not target_python.exists():
            _create_venv(target_python)

    _install(target_python)
    _relaunch(target_python)
