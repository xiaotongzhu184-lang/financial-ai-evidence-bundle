#!/usr/bin/env python3
"""Run the unittest suite with the same native-runtime setup as the CLI."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def test_environment() -> dict[str, str]:
    environment = os.environ.copy()
    if sys.platform != 'darwin':
        return environment
    spec = importlib.util.find_spec('sklearn')
    if spec is None or not spec.origin:
        return environment
    directory = Path(spec.origin).parent / '.dylibs'
    if not (directory / 'libomp.dylib').is_file():
        return environment
    existing = environment.get('DYLD_LIBRARY_PATH', '')
    environment['DYLD_LIBRARY_PATH'] = str(directory) + (os.pathsep + existing if existing else '')
    return environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', default='test*.py',
                        help='unittest discovery pattern; default: test*.py')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, '-m', 'unittest', 'discover',
               '-s', str(root / 'tests'), '-p', args.pattern, '-v']
    return subprocess.run(command, cwd=root, env=test_environment()).returncode


if __name__ == '__main__':
    raise SystemExit(main())
