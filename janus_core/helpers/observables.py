"""
Module for built-in correlation observables.
"""

from ase import Atoms, units
from typing import Dict

class ComponentMixin:
    """
    Mixin for handling Observables with components.

    Parameters
    ----------
    component : str
        Symbol for observables component.
    """
    def __init__(self, component: str):
        self._check_component(component)
        self.component = component
        self._index = self._components[self.component]
    
    @property
    def _components(self) -> Dict[str, int]:
        """
        Allowed symbolic components with associated indices.

        Returns
        -------
        Dict[str, int]
            The allowed components and associated indices.
        """
        return {}
    
    def _check_component(self, component):
        """
        Check if a component is valid.

        Raises
        ------
        ValueError
            If the component is invalid.
        """
        if component not in self._components:
            raise ValueError(
                f"'{component}' invalid, must be '{', '.join(list(self._components.keys()))}'"
            )

# pylint: disable=too-few-public-methods
class Stress(ComponentMixin):
    """
    Observable for stress components.

    Parameters
    ----------
    component : str
        Symbol for tensor components, xx, yy, etc.
    include_ideal_gas : bool
        Calculate with the ideal gas contribution.
    """

    @property
    def _components(self) -> Dict[str, int]:
        return {
            "xx": 0,
            "yy": 1,
            "zz": 2,
            "yz": 3,
            "zy": 3,
            "xz": 4,
            "zx": 4,
            "xy": 5,
            "yx": 5}

    def __init__(self, component: str, *, include_ideal_gas: bool = True):
        """
        Initialise the observable from a symbolic str component.

        Parameters
        ----------
        component : str
            Symbol for tensor components, xx, yy, etc.
        include_ideal_gas : bool
            Calculate with the ideal gas contribution.
        """
        super().__init__(component)
        self.include_ideal_gas = include_ideal_gas

    def __call__(self, atoms: Atoms, *args, **kwargs) -> float:
        """
        Get the stress component.

        Parameters
        ----------
        atoms : Atoms
            Atoms object to extract values from.
        *args : tuple
            Additional positional arguments passed to getter.
        **kwargs : dict
            Additional kwargs passed getter.

        Returns
        -------
        float
            The stress component in GPa units.
        """
        return (
            atoms.get_stress(include_ideal_gas=self.include_ideal_gas, voigt=True)[
                self._index
            ]
            / units.GPa
        )


# pylint: disable=too-few-public-methods
class Velocity(ComponentMixin):
    """
    Observable for per atom velocity components.

    Parameters
    ----------
    component : str
        Symbol for velocity components, x, y, z.
    atom : int
        Atom index
    """

    @property
    def _components(self) -> Dict[str, int]:
        return {"x": 0, "y": 1, "z": 2}

    def __init__(self, component: str, *, atom: int):
        """
        Initialise the observable from a symbolic str component and atom index. 

        Parameters
        ----------
        component : str
            Symbol for tensor components, xx, yy, etc.
        include_ideal_gas : bool
            Calculate with the ideal gas contribution.
        """
        super().__init__(component)
        self._atom_index = atom

    def __call__(self, atoms: Atoms, *args, **kwargs) -> float:
        """
        Get the velocity component of an atom.

        Parameters
        ----------
        atoms : Atoms
            Atoms object to extract values from.
        *args : tuple
            Additional positional arguments passed to getter.
        **kwargs : dict
            Additional kwargs passed getter.

        Returns
        -------
        float
            The velocity component.
        """
        return (
            atoms.get_velocities()[self._atom_index, self._index]
        )
