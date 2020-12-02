import sys
import subprocess


def pip_install(pkg):
    if pkg not in sys.modules:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
