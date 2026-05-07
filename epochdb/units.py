from typing import Optional, Union
import pint

class UnitRegistry:
    """Wrapper around pint for dimensional analysis and unit conversion."""
    def __init__(self):
        self.ureg = pint.UnitRegistry()
        # Ensure we have a consistent state
        self.ureg.define('USD = [currency]')
        self.ureg.define('EUR = 1.08 * USD')
        self.ureg.define('BRL = 0.20 * USD')

    def parse(self, value: float, unit: str):
        """Creates a Quantity object."""
        try:
            return self.ureg.Quantity(value, unit)
        except:
            return value * self.ureg.dimensionless

    def compatible(self, unit1: str, unit2: str) -> bool:
        """Check if two units are dimensionally consistent."""
        try:
            q1 = self.ureg.Quantity(1, unit1)
            q2 = self.ureg.Quantity(1, unit2)
            return q1.is_compatible_with(q2)
        except:
            return False

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert value from one unit to another."""
        try:
            q = self.ureg.Quantity(value, from_unit)
            return q.to(to_unit).magnitude
        except:
            return value

    def get_base_dimensions(self, unit: str) -> str:
        """Returns the base dimensions of a unit (e.g., [mass] * [length] / [time]**2)."""
        try:
            q = self.ureg.Quantity(1, unit)
            return str(q.dimensionality)
        except:
            return "dimensionless"
