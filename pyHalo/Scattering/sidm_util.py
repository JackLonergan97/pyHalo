from pyHalo.Scattering.isothermal_jeans import compute_r1, solve_iterative, integrate_profile
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Cosmology.cosmology import Cosmology
from scipy.interpolate import interp1d
from pyHalo.Halos.halo_util import deflection_angle
import numpy as np
from pyHalo.Scattering.sidm_interp import logrho
from pyHalo.Scattering.cross_sections import VelocityDependentCross

cosmo = Cosmology()
lens_cosmo = LensCosmo(0.5, 1.5, cosmo)
lens_cosmo.rhoc *= 0.7 ** 2

def sidm_central_density_from_mass_exact(c_norm, vpower, m, z, N=5, plot=False):

    thalo = cosmo.halo_age(z)

    rhonfw, rs_nfw, _ = lens_cosmo.NFW_params_physical_fromM(m, z)

    rho0, s0, core_size_unitsrs, fit_quality = \
        solve_iterative(rhonfw, rs_nfw, c_norm, vpower, thalo, N, plot)
    return rho0, rhonfw/rho0, rs_nfw * rhonfw/rho0, s0

def sidm_central_density_from_mass_interpolated(c_norm, vpower, m, z, c_scatter=False):

    halo_age_scaled = cosmo.halo_age(z) / 10
    zeta = c_norm * halo_age_scaled
    rhos, rs, _ = lens_cosmo.NFW_params_physical_fromM(m, z)

    delta_c_dex = 0.
    logrho0 = logrho(np.log10(m), z, zeta, vpower, delta_c_dex)
    rho0 = 10**logrho0
    return rho0, rhos/rho0, rs * rhos/rho0

def nfw_density_truncated(x, tau, rhos):
    return rhos * tau ** 2 / (x * (1 + x) ** 2) / (x ** 2 + tau ** 2)

class Profile(object):

    def __init__(self, inner_domain_x, inner_density_function, outer_density_function):

        self._inner_domain_x = inner_domain_x
        self._xmin = self._inner_domain_x[0]
        self._xmax = self._inner_domain_x[-1]

        self._inner = inner_density_function
        self._outer = outer_density_function

        self._matching_rho = self._inner(self._xmax)

    def eval_point(self, x, args):

        x = abs(x)

        if x <= self._xmax:
            return self._inner(max(x, self._xmin))
        else:
            norm = self._matching_rho / self._outer(self._xmax, *args)
            return self._outer(x, *args) * norm

    def __call__(self, x, args):

        if isinstance(x, float) or isinstance(x, int):

            return self.eval_point(x, args)

        else:

            values = [self.eval_point(xi, args) for xi in x]
            return np.array(values)


class CompositeSIDMProfile(object):

    def __init__(self, rho_s, r_s, cross_section_norm, v_power, t_halo, tau=10):

        rho0, vdis_isothermal, core_density_ratio, _ = solve_iterative(rho_s, r_s, cross_section_norm, v_power, t_halo,
                                                                       6, plot=False,
                                                                       tol=0.01)
        r_1 = compute_r1(rho_s, r_s, vdis_isothermal, cross_section_norm, v_power, t_halo,)
        domain_r, rhoiso = integrate_profile(rho0, vdis_isothermal, r_s, r_1)
        domain_x_inner = domain_r[0:-1] / r_s

        self.rho0 = rho0
        self.rho_s = rho_s
        self.r_s = r_s
        self.r_1 = r_1
        self.tau = tau
        self.vdis_isothermal = vdis_isothermal
        self.core_density_ratio = core_density_ratio
        self._args = (self.tau, self.rho_s)
        self._profile = Profile(domain_x_inner, interp1d(domain_x_inner, rhoiso[0:-1]), nfw_density_truncated)

    @classmethod
    def from_mass(cls, M, z, cross_section, tau=10, add_c_scatter=None):

        c = lens_cosmo.NFW_concentration(M, z, scatter=False)

        if add_c_scatter is not None:
            c = 10**(np.log10(c) + add_c_scatter)

        rho_s, r_s, _ = lens_cosmo.NFW_params_physical(M, c, z)
        composite_profile = cls(rho_s, r_s, cross_section, tau)
        return composite_profile

    def __call__(self, x):
        return self._profile(x, (self.tau, self.rho_s))

    def deflection_angle(self, x, zmax_factor=500):

        if isinstance(x, float) or isinstance(x, int):
            def_angle = deflection_angle(x, self, (), zmax_factor*self.r_s)
        else:
            def_angle = [deflection_angle(xi, self, (), zmax_factor*self.r_s) for xi in x]
            def_angle = np.array(def_angle)
        return def_angle

