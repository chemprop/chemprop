import inspect
from typing import Any, Collection


class ReprMixin:
    def __repr__(self) -> str:
        """Return a string representation of the object."""

        param_repr_list = []
        for param_name, param_value in self.get_params(deep=False).items():
            param_default = self._get_default_params()[param_name]
            if param_value != param_default:
                param_repr_list.append(f"{param_name}={repr(param_value)}")

        argspec = ", ".join(param_repr_list)
        return f"{self.__class__.__name__}({argspec})"

    def _get_default_params(self) -> dict[str, Any]:
        """Get the default parameters for initializing the class."""
        sig = inspect.signature(self.__class__)
        return {k: v.default for k, v in sig.parameters.items()}

    def _get_parm_names(self) -> list[str]:
        """Get the default parameters for initializing the class."""
        return sorted(self._get_default_params().keys())

    def get_params(self, deep=True):
        """Get the current parameters of the class, required to initialize the class.

        This method mimics the behavior of `sklearn.base.BaseEstimator.get_params`.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for param_name, _ in self._get_parm_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out


