from typing import cast

import numpy
from numpy.typing import NDArray


def step_function(
    x: NDArray[numpy.float64],
) -> NDArray[numpy.float64]:
    y = x > 0
    return cast(NDArray[numpy.float64], y.astype(numpy.int_))
