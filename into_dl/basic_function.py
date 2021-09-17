import warnings
from typing import cast

import numpy
from numpy.typing import NDArray


def step_function(
    x: NDArray[numpy.float64],
) -> NDArray[numpy.float64]:
    y = x > 0
    return cast(NDArray[numpy.float64], y.astype(numpy.float64))


def sigmoid(
    x: NDArray[numpy.float64],
) -> NDArray[numpy.float64]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return 1.0 / (1.0 + numpy.exp(-x))
