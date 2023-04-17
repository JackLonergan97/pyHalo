import numpy as np
from copy import deepcopy
from pyHalo.Rendering.rendering_class_base import RenderingClassBase


class LineOfSightNoSheet(RenderingClassBase):
    """
    This class generates line-of-sight halos, or more precisely objects between the observer and the source that are
    not associated with the host dark matter halo around the main deflector.
    """

    def __init__(self, mass_function_model, kwargs_mass_function, spatial_distribution_model,
                 geometry, lens_cosmo, lens_plane_redshifts, delta_z_list):

        """

        :param keywords_master: a dictionary of keyword arguments to be passed to each model class
        :param lens_cosmo: an instance of LensCosmo (see Halos.lens_cosmo)
        :param geometry: an instance of Geometry (see Cosmology.geometry)
        :param halo_mass_function: an instance of LensingMassFunction (see Cosmology.lensing_mass_function)
        :param lens_plane_redshifts: a list of redshifts at which to render halos
        :param delta_z_list: a list of redshift increments between each lens plane (should be the same length as
        lens_plane_redshifts)
        """
        self._mass_function_model = mass_function_model
        self._spatial_distribution_model = spatial_distribution_model
        self._kwargs_mass_function = kwargs_mass_function
        self._geometry = geometry
        self._lens_cosmo = lens_cosmo
        self._lens_plane_redshifts = lens_plane_redshifts
        self._delta_z_list = delta_z_list
        super(LineOfSightNoSheet, self).__init__()

    def render(self):

        """
        Generates halo masses and positions for objects along the line of sight
        (except for halos from the two-halo contribution)
        :return: mass (in Msun), x (arcsec), y (arcsec), r3d (kpc), redshift
        """

        masses = np.array([])
        x = np.array([])
        y = np.array([])
        redshifts = np.array([])
        first_mass_function_moment = []
        z_mass_function_moment = []

        for z, dz in zip(self._lens_plane_redshifts, self._delta_z_list):
            mfunc_model = self._get_mass_function_model(z, dz)
            m = mfunc_model.draw()
            z_mass_function_moment.append(z)
            first_mass_function_moment.append(mfunc_model.first_moment)
            nhalos = len(m)
            _x, _y = self.render_positions_at_z(z, nhalos)
            _z = np.array([z] * len(_x))
            masses = np.append(masses, m)
            x = np.append(x, _x)
            y = np.append(y, _y)
            redshifts = np.append(redshifts, _z)

        subhalo_flag = [False] * len(masses)
        r3d = np.array([None] * len(masses))
        return masses, x, y, r3d, redshifts, subhalo_flag

    def render_positions_at_z(self, z, nhalos):

        """
        :param z: redshift
        :param nhalos: number of halos or objects to generate
        :return: the x, y coordinate of objects in arcsec, and a 3 dimensional coordinate in kpc
        The 3d coordinate only has a clear physical interpretation for subhalos, and is used to compute truncation raddi.
        For line of sight halos it is set to None.
        """

        x_kpc, y_kpc = self._spatial_distribution_model.draw(nhalos, z)

        if len(x_kpc) > 0:
            kpc_per_asec = self._geometry.kpc_per_arcsec(z)
            x_arcsec = x_kpc * kpc_per_asec ** -1
            y_arcsec = y_kpc * kpc_per_asec ** -1
            return x_arcsec, y_arcsec

        else:
            return np.array([]), np.array([])

    def _get_mass_function_model(self, z, delta_z, log_mlow=None, log_mhigh=None):
        """

        :param z:
        :param delta_z:
        :param log_mlow: replaces log_mlow in kwargs_mass_function if not None
        :param log_mhigh: replaces log_mhigh in kwargs_mass_function if not None
        :return:
        """
        kwargs_model = deepcopy(self._kwargs_mass_function)
        if log_mlow is not None:
            kwargs_model['log_mlow'] = log_mlow
        if log_mhigh is not None:
            kwargs_model['log_mhigh'] = log_mhigh
        line_of_sight_rescaling = kwargs_model['LOS_normalization']
        del kwargs_model['LOS_normalization']
        if 'delta_power_law_index' in kwargs_model.keys():
            delta_power_law_index = kwargs_model['delta_power_law_index']
            del kwargs_model['delta_power_law_index']
        else:
            delta_power_law_index = 0.0
        mfunc_model = self._mass_function_model.from_redshift(z, delta_z, self._geometry, line_of_sight_rescaling,
                                                              kwargs_model, delta_power_law_index)
        return mfunc_model

    def convergence_sheet_correction(self, kappa_scale, log_mlow, log_mhigh,
                                     kwargs_mass_sheets=None, zmin=None, zmax=None):
        return [{}], [], []


class LineOfSight(LineOfSightNoSheet):

    """
    This class generates line-of-sight halos, or more precisely objects between the observer and the source that are
    not associated with the host dark matter halo around the main deflector.
    """

    def convergence_sheet_correction(self, kappa_scale, log_mlow, log_mhigh,
                                     kwargs_mass_sheets=None, zmin=None, zmax=None):

        """
        this routine applies the negative convergence sheet correction for lens planes along the line of sight
        :param kwargs_mass_sheets: keyword arguments that overwrite whatever the default settings for the mass sheet
        sheet are - leave it as None for most applications
        :return:
        """

        lens_plane_redshifts = self._lens_plane_redshifts[0::2]
        delta_zs = 2 * self._delta_z_list[0::2]
        kwargs_out = []
        profile_names_out = []
        redshifts = []

        if zmin is None:
            zmin = 0.0
        if zmax is None:
            zmax = self._geometry.zsource

        for z, delta_z in zip(lens_plane_redshifts, delta_zs):

            if z < zmin:
                continue
            if z > zmax:
                continue

            log_mass_sheet_correction_min, log_mass_sheet_correction_max = self._redshift_dependent_mass_range(
                z, log_mlow, log_mhigh)
            mass_function_model = self._get_mass_function_model(z, delta_z,
                                                                log_mlow=log_mass_sheet_correction_min,
                                                                log_mhigh=log_mass_sheet_correction_max)
            first_moment = mass_function_model.first_moment
            kappa = self._convergence_at_z(z, self._geometry, self._lens_cosmo, kappa_scale, first_moment)
            if kappa > 0:
                kwargs_out.append({'kappa': -kappa})
                profile_names_out += ['CONVERGENCE']
                redshifts.append(z)
        return kwargs_out, profile_names_out, redshifts

    @staticmethod
    def _convergence_at_z(z, geometry_class, lens_cosmo_class, kappa_scale, mtheory):

        area = geometry_class.angle_to_physical_area(0.5 * geometry_class.cone_opening_angle, z)
        sigma_crit_mass = lens_cosmo_class.sigma_crit_mass(z, area)
        return kappa_scale * mtheory / sigma_crit_mass
