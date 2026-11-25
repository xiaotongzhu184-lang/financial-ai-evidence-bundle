"""Process-local macOS OpenMP setup for the installed CLI, including wheels."""
import importlib.util
import os
from pathlib import Path
import sys


def prepare_cli_runtime() -> None:
    """Restart once with sklearn's bundled OpenMP directory before native imports.

    exec preserves foreground signal handling: Ctrl-C reaches the actual server,
    rather than interrupting a launcher waiting on a child process. No system
    libraries or persistent environment settings are changed.
    """
    if sys.platform != 'darwin' or os.environ.get('YK_EVIDENCE_OPENMP_READY') == '1':
        return
    if '--help' in sys.argv or '-h' in sys.argv:
        return
    spec = importlib.util.find_spec('sklearn')
    if spec is None or not spec.origin:
        return
    directory = Path(spec.origin).parent / '.dylibs'
    if not (directory / 'libomp.dylib').is_file():
        return
    environment = os.environ.copy()
    existing = environment.get('DYLD_LIBRARY_PATH', '')
    environment['DYLD_LIBRARY_PATH'] = str(directory) + (os.pathsep + existing if existing else '')
    environment['YK_EVIDENCE_OPENMP_READY'] = '1'
    os.execve(sys.executable, [sys.executable, '-m', 'yk_evidence', *sys.argv[1:]], environment)
