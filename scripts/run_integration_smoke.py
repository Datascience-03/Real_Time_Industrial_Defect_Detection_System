import sys
import os
import traceback
import importlib.util

# ensure repository root is on sys.path so `import src` works
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


smoke1 = load_module("test_smoke_imports", "tests/integration/test_smoke_imports.py")
smoke2 = load_module("test_api_endpoints", "tests/integration/test_api_endpoints.py")

tests = [
    ("import_src_app", smoke1.test_import_src_app),
    ("realtime_help", smoke1.test_realtime_help),
    ("home_and_version", smoke2.test_home_and_version),
    ("health", smoke2.test_health),
    ("predict_endpoint", smoke2.test_predict_endpoint_behaviour),
]

failed = 0

for name, fn in tests:
    try:
        print(f"RUNNING: {name}")
        fn()
        print(f"OK: {name}\n")
    except AssertionError as e:
        print(f"FAIL: {name} - AssertionError: {e}\n")
        traceback.print_exc()
        failed += 1
    except Exception as e:
        print(f"ERROR: {name} - Exception: {e}\n")
        traceback.print_exc()
        failed += 1

if failed:
    print(f"{failed} tests failed")
    sys.exit(2)
else:
    print("All integration smoke tests passed")
    sys.exit(0)
