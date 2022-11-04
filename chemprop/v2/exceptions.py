from typing import Iterable


class InvalidShapeError(ValueError):
    def __init__(self, var_name: str, received: Iterable[int], expected: Iterable[int]):
        message = (
            f"arg '{var_name}' has incorrect shape! "
            f"got: `{self.pretty_shape(received)}`. expected: `{self.pretty_shape(expected)}`"
        )
        super().__init__(message)

    @classmethod
    def pretty_shape(cls, shape: Iterable[int]) -> str:
        """Make a pretty string from an input shape

        Example
        --------
        >>> X = np.random.rand(10, 4)
        >>> X.shape
        (10, 4)
        >>> InvalidShapeError.pretty_shape(X.shape)
        '10 x 4'
        """
        return " x ".join(map(str, shape))
