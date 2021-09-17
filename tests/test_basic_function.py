import numpy

from into_dl.basic_function import step_function


def test_step_function() -> None:
    y = step_function(numpy.array([1.0, 10.0]))
    assert numpy.all(y > 0) and y.shape == (2,)
    y = step_function(numpy.array([-1.0, -10.0, 0.0]))
    assert numpy.all(y == 0) and y.shape == (3,)
