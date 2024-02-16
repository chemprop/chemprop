"""Mixin classes for the project."""

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

        non_default_params = ", ".join(param_repr_list)
        return f"{self.__class__.__name__}({non_default_params})"

    @classmethod
    def _get_default_params(cls) -> dict[str, Any]:
        """Get the default parameters for initializing the class."""
        sig = inspect.signature(cls)
        return {k: v.default for k, v in sig.parameters.items()}

    def _get_parm_names(self) -> list[str]:
        """Get the default parameters for initializing the class."""
        return sorted(self._get_default_params().keys())

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get the current parameters of the class, required to initialize the class.

        This method mimics the behavior of `sklearn.base.BaseEstimator.get_params`.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        dict[str, Any]
            Parameter names mapped to their values.
        """
        out: dict[str, Any] = {}
        for param_name in self._get_parm_names():
            value = getattr(self, param_name)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((param_name + "__" + k, val) for k, val in deep_items)
            out[param_name] = value
        return out
