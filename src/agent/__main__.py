from  __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent import config


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent")
    parser.add_argument("--print-config", action="store_true",
                        help="show every setting and where it came from")
    parser.add_argument("--model")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--permission-mode", dest="permission_mode",
                        choices=["readonly", "ask", "auto", "full"])
    args = parser.parse_args()

    flags = {
        "model": args.model,
        "effort": args.effort,
        "permission_mode": args.permission_mode,
    }

    try:
        resolved = config.load(Path.cwd(), flags)
    except config.ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        raise SystemExit(2)

    if args.print_config:
        print(config.render(resolved))
        return

    print("nothing to do yet — try --print-config")
