import numpy as np
import json
import sys

from gs_runner import gs_runner


if __name__ == "__main__":
    json_path = sys.argv[1]
    runner = gs_runner(json_path=json_path)
    runner.run()
    runner.close()
