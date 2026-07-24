import numpy.testing as npt
import numpy as np
import pytest
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Halos.HaloModels.globular_cluster import GlobularClusterKing


class TestGlobularClusters(object):

    def setup_method(self):
        self.zhalo = 0.4
        self.zsource = 2.0
        self.lens_cosmo = LensCosmo(self.zhalo, self.zsource, None)

    def test_lenstronomy_ID(self):
        mass = 10 ** 5
        args = {'r_h': 2.5, 'c': 2.0}
        profile = GlobularClusterKing(mass, 0.0, 0.0, self.zhalo, self.lens_cosmo,
                                      args, 1)
        lenstronomy_ID = profile.lenstronomy_ID
        npt.assert_string_equal(lenstronomy_ID[0], 'KING')

    def test_lenstronomy_args(self):
        mass = 10 ** 5.0
        args = {'r_h': 2.5, 'c': 2.0}
        profile = GlobularClusterKing(mass, 0.0, 0.0, self.zhalo, self.lens_cosmo,
                                      args, 1)
        lenstronomy_args, delta_z = profile.lenstronomy_params
        assert delta_z is None
        kw = lenstronomy_args[0]
        for key in ['sigma0', 'r_h', 'c', 'center_x', 'center_y']:
            assert key in kw
        # r_h carried through to arcsec, c passed straight through
        kpc_per_arcsec = profile.lens_cosmo.cosmo.kpc_proper_per_asec(profile.z)
        npt.assert_almost_equal(kw['r_h'], 1e-3 * args['r_h'] / kpc_per_arcsec)
        npt.assert_almost_equal(kw['c'], args['c'])

    def test_mass(self):
        logM = 6.0
        mass = 10 ** logM
        args = {'r_h': 3.0, 'c': 1.5}
        profile = GlobularClusterKing(mass, 0.0, 0.0, self.zhalo, self.lens_cosmo,
                                      args, 1)

        # profile_args = (sigma0 [Msun/pc^2], r_h [pc], c)
        sigma0_pc, r_h_pc, c = profile.profile_args
        npt.assert_almost_equal(r_h_pc, args['r_h'])
        npt.assert_almost_equal(c, args['c'])

        # King is a 2D-only profile: total *projected* mass == input mass
        total_mass = profile._prof.mass_2d_lens(1e10, sigma0_pc, r_h_pc, c)[0]
        npt.assert_almost_equal(total_mass / mass, 1.0, 6)

        # r_h really is the projected half-mass radius
        half_mass = profile._prof.mass_2d_lens(r_h_pc, sigma0_pc, r_h_pc, c)[0]
        npt.assert_almost_equal(half_mass / total_mass, 0.5, 3)

        # same check in lenstronomy (convergence) units
        kpc_per_arcsec = profile.lens_cosmo.cosmo.kpc_proper_per_asec(profile.z)
        kw = profile.lenstronomy_params[0][0]
        sigma0, r_h_arcsec, c = kw['sigma0'], kw['r_h'], kw['c']
        sigma_crit_mpc = profile.lens_cosmo.get_sigma_crit_lensing(
            profile.z, profile.lens_cosmo.z_source)
        sigma_crit_arcsec = sigma_crit_mpc * (0.001 * kpc_per_arcsec) ** 2  # Msun/arcsec^2
        m2d_conv = profile._prof.mass_2d_lens(1e10, sigma0, r_h_arcsec, c)[0]
        total_mass_ls = m2d_conv * sigma_crit_arcsec
        npt.assert_almost_equal(total_mass_ls / mass, 1.0, 4)

        # integrate the projected surface density out to the tidal radius r_t
        r_core_pc = float(profile._prof._r_core(r_h_pc, c))
        r_t_pc = r_core_pc * 10 ** c
        r = np.linspace(1e-5, 1.0, 200000) * r_t_pc
        sigma = profile.density_profile_2d_lenstronomy(r)   # Msun/pc^2 at r [pc]
        m2d = np.trapezoid(2 * np.pi * r * sigma, r)
        npt.assert_almost_equal(m2d / mass, 1.0, 3)


if __name__ == '__main__':
    pytest.main()
