import subprocess
import sys


def test_import_src_app():
    # ensure the package imports without error
    subprocess.check_call([sys.executable, "-c", "import src.app"])


def test_realtime_help():
    # ensure realtime_inference CLI is reachable and prints help
    subprocess.check_call([sys.executable, "realtime_inference.py", "--help"])
