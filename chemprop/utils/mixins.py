import inspect
from typing import Any, Collection


class ReprMixin:
    def __repr__(self) -> str:
        default_params = self._get_default_params()
        param_repr_list = [
            f"{name}={repr(value)}"
            for name, value in self.get_params(deep=False).items()
            if value != default_params[name]
        ]

        return f"{self.__class__.__name__}({','.join(param_repr_list)})"

    @classmethod
    def _get_default_params(cls) -> dict[str, Any]:
        """Get the default parameters for initializing the class."""
        sig = inspect.signature(cls)
        return {k: v.default for k, v in sig.parameters.items()}

    def _get_param_names(self) -> list[str]:
        """Get the parameter names for initializing the class."""
        return sorted(self._get_default_params().keys())

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get the current parameters of the class, required to initialize the class.

        This method mimics the behavior of :method:`sklearn.base.BaseEstimator.get_params`.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this object and all contained subobjects
            which implement the `get_params` method.

        Returns
        -------
        dict[str, Any]
             The parameter names and respective values used to initialize this class.
             That is, an identical initialization of this object can be achieved via:
             ```python
                o: ReprMixin
                o_copy = type(o)(**o.get_params(deep=False))
            ```
        """
        name_to_value = {}
        for name in self._get_default_params.keys():
            value = getattr(self, name)
            name_to_value[name] = value
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                _params = value.get_params()
                name_to_value.update((f"{name}__{k}", v) for k, v in _params.items())

        return name_to_value
