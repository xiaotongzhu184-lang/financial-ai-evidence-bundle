#!/usr/bin/env python3
"""Compatibility launcher; runtime setup now lives in the installed CLI."""
import os
import subprocess
import sys


def main():
    # Windows emulates exec rather than replacing this process. Wait explicitly
    # there so callers receive the verifier's nonzero exit status.
    if os.name == 'nt':
        return subprocess.run([sys.executable, '-m', 'yk_evidence', *sys.argv[1:]]).returncode
    os.execv(sys.executable, [sys.executable, '-m', 'yk_evidence', *sys.argv[1:]])


if __name__ == '__main__':
    sys.exit(main())
