"""vlt-claude — thin transparent wrapper around the claude CLI.

Passthrough (os.execv, zero remaining overhead):
  - VLT_RELAY_ACTIVE is set   → recursion guard
  - stdin is not a TTY        → piped/scripted input
  - -p / --print in args      → SDK/batch mode
  - first arg is a subcommand → version, help, update, mcp, config …

Interactive TTY → PTY relay via vlt.commands.session_relay.run_relay().
"""
import os
import sys

_PASS_FLAGS = frozenset(["-p", "--print", "--version", "--help", "-h"])
_PASS_CMDS = frozenset(
    ["version", "help", "update", "upgrade", "mcp", "config", "doctor", "bug", "api"]
)


def main() -> None:
    args = sys.argv[1:]

    if (
        os.environ.get("VLT_RELAY_ACTIVE")         # recursion guard
        or not sys.stdin.isatty()                   # piped stdin
        or any(a in _PASS_FLAGS for a in args)      # SDK / special flags
        or (args and args[0] in _PASS_CMDS)         # non-interactive subcommands
    ):
        _exec_real(args)

    # Interactive relay mode — lazy import keeps hot-path startup minimal
    os.environ["VLT_RELAY_ACTIVE"] = "1"
    from vlt.commands.session_relay import run_relay
    sys.exit(run_relay(args))


def _exec_real(args: list[str]) -> None:
    from vlt.commands.session_relay import find_real_claude
    real = find_real_claude()
    if sys.platform == "win32":
        import subprocess
        sys.exit(subprocess.call([real] + args))
    os.execv(real, [real] + args)
