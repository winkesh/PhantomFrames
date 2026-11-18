import os
import sys

# --res must be known before anything imports phantom.config (resolution is fixed at import time). The engine's
# worker processes inherit PHANTOM_RES from this environment.
if "--res" in sys.argv:
    i = sys.argv.index("--res")
    if i + 1 < len(sys.argv):
        os.environ["PHANTOM_RES"] = sys.argv[i + 1]

if __name__ == "__main__":      # worker processes re-import this module (spawn): only the parent runs the app
    from .server import main

    main()
