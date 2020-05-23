import numpy as np
import json
import sys

from gs_runner_LSTM import gs_runner_LSTM
from gs_runner_DENSE import gs_runner_DENSE


if __name__ == "__main__":
    json_path = sys.argv[1]
    runner = gs_runner_DENSE(json_path = json_path)
    runner.run()

    print('ciao')