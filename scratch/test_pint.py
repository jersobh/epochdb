import pint
ureg = pint.UnitRegistry()
try:
    q1 = 1 * ureg("degC")
    q2 = 1 * ureg("degC")
    print(f"degC compatible with degC: {q1.is_compatible_with(q2)}")
except Exception as e:
    print(f"Error: {e}")
