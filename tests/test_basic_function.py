import numpy

from into_dl.basic_function import sigmoid, step_function


def test_step_function() -> None:
    y = step_function(numpy.array([1.0, 10.0]))
    assert numpy.all(y > 0) and y.shape == (2,)
    y = step_function(numpy.array([-1.0, -10.0, 0.0]))
    assert numpy.all(y == 0) and y.shape == (3,)


def test_sigmoid() -> None:
    y = sigmoid(numpy.array([0.0]))
    assert y == numpy.array([0.5])

    z = 1.0e5
    y = sigmoid(numpy.array([z, -z]))
    assert 0.9 < y[0] <= 1.0
    assert 0 <= y[1] < 0.1
