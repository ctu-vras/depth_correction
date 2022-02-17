from __future__ import absolute_import, division, print_function
# from enum import Enum
import numpy as np
import torch

__all__ = [
    'beam_radius',
    'Medium',
    'Media',
    'rayleight_length',
    'Sensor',
    'Sensors',
]


class Medium(object):
    def __init__(self, refractive_index=None):
        self.refractive_index = refractive_index


# class Media(Enum):
class Media(object):
    AIR = Medium(refractive_index=1.000293)
    VACUUM = Medium(refractive_index=1.0)


def rayleight_length(waist_radius, wavelength, n=Media.AIR.refractive_index):
    """Rayleight lenght (range) for given beam waist.

    :param waist_radius: beam waist [m].
    :param n: index of refraction of the propagation medium,
            n=1.0 for vacuum, n=1.000293 for air (default).
    :param
    """
    if not isinstance(waist_radius, torch.Tensor):
        waist_radius = torch.as_tensor(waist_radius)
    assert isinstance(waist_radius, torch.Tensor)

    z_r = torch.pi * waist_radius**2 * n / wavelength
    return z_r


def beam_radius(z, waist_radius, wavelength, m2, n=Media.AIR.refractive_index):
    """Beam width at given depth.

    :param z: depth [m].
    :param waist_radius: beam radius [m].
    :param n: index of refraction of the propagation medium.
    :param wavelength: wavelength [m].
    :param m2: M2, "M squared", beam quality factor, or beam propagation factor.
            m2=1.0 for ideal Gaussian beam (default).
    :return: Beam width [m].
    """
    if not isinstance(z, torch.Tensor):
        z = torch.as_tensor(z)
    assert isinstance(z, torch.Tensor)

    if not isinstance(waist_radius, torch.Tensor):
        waist_radius = torch.as_tensor(waist_radius)
    assert isinstance(waist_radius, torch.Tensor)

    w = waist_radius * m2 * torch.sqrt(1.0 + (z / rayleight_length(waist_radius, wavelength=wavelength, n=n))**2)
    return w


def beam_propagation_factor(divergence, waist_radius, wavelength):
    return divergence * torch.pi * waist_radius / wavelength


class Sensor(object):
    def __init__(self, name=None, wavelength=None, waist_radius=None, divergence=None, m2=1.0):
        self.name = name
        self.wavelength = wavelength
        self.waist_radius = waist_radius
        self.divergence = divergence
        if divergence is not None:
            self.m2 = self.beam_propagation_factor()
        else:
            self.m2 = m2

    def rayleight_length(self, n=Media.AIR.refractive_index):
        return rayleight_length(self.waist_radius, n=n, wavelength=self.wavelength)

    def beam_radius(self, z, n=Media.AIR.refractive_index):
        return beam_radius(z, waist_radius=self.waist_radius, n=n, wavelength=self.wavelength, m2=self.m2)

    def beam_propagation_factor(self):
        return beam_propagation_factor(self.divergence, self.waist_radius, self.wavelength)

    def __str__(self):
        return self.name


# class Sensors(Enum):
class Sensors(object):
    # TODO: Hokuyo waist_radius and divergence from Ouster
    HOKUYO = Sensor(name='Hokuyo UTM-30LX', wavelength=905e-9, waist_radius=5e-3 / 2, divergence=np.radians(0.35))
    # TODO: Ouster waist_radius=5e-3 / 2 FWHM?
    OUSTER = Sensor(name='Ouster OS0', wavelength=865e-9, waist_radius=5e-3 / 2, divergence=np.radians(0.35))


def demo():
    for sensor in [Sensors.HOKUYO, Sensors.OUSTER]:
        print(sensor)
        print('Rayleight length: %.3f m' % sensor.rayleight_length())
        print('Beam propagation factor: %.3f' % sensor.beam_propagation_factor())
        for z in [0.0, 0.5, 1.0, 5.0, 10.0]:
            print('Beam radius at %.3f m: %.6f m' % (z, sensor.beam_radius(z)))
        print()


def main():
    demo()


if __name__ == '__main__':
    main()
