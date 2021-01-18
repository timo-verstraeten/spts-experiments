import numpy as np

def compute_Cp_Ct(air_density, swept_area, wind_speed, power, thrust):
    factor = 0.5 * air_density * swept_area * wind_speed ** 2
    Cp = power / factor / wind_speed * 1000
    Ct = thrust / factor * 1000
    return Cp, Ct

def compute_swept_area(rotor_diameter):
    return np.pi * (rotor_diameter / 2)**2