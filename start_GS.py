import numpy as np
import json
import sys

from models.gs_runner_LSTM import gs_runner_LSTM
from models.gs_runner_DENSE import gs_runner_DENSE
from models.gs_runner_LSTM_onestep import gs_runner_LSTM_onestep


if __name__ == "__main__":
    json_path = sys.argv[1]
    runner = gs_runner_LSTM(json_path = json_path)
    runner.run()
    runner.close()
