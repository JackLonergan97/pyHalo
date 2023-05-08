from abc import ABC, abstractmethod
import numpy as np
import inspect
from pyHalo.utilities import ITSampling
_path_galacticus_data_testing = inspect.getfile(inspect.currentframe())[0:-13]+'/adiabatic_tides_data/galacticus_data.txt'
_path_galacticus_data_run = inspect.getfile(inspect.currentframe())[0:-8]+'/adiabatic_tides_data/galacticus_data.txt'
try:
    _rperi_sampling = ITSampling(np.loadtxt(_path_galacticus_data_run)[:,0])
except:
    _rperi_sampling = ITSampling(np.loadtxt(_path_galacticus_data_testing)[:,0])

class Halo(ABC):

    def __init__(self, mass=None, x=None, y=None, r3d=None, mdef=None, z=None,
                 sub_flag=None, lens_cosmo_instance=None, args={}, unique_tag=None, fixed_position=False):

        """
        This is the main class for objects rendered in the lens volume. It keeps track of stuff like the position,
        mass, redshift, and structural properties (e.g. concentration, core radius, etc.)

        :param mass: halo mass in M_sun
        :param x: angular coordinate x in arcsec
        :param y: angular coordinate y in arcsec
        :param r3d: 3D position of halo in kpc (used to compute the truncation radius for subhalos)
        For field halos this is not relevant and is set to None
        :param mdef: mass definition for the halo
        :param z: halo redshift
        :param sub_flag: bool; if True, the halo is treated as a main deflector subhalo
        :param lens_cosmo_instance: an instance of LensCosmo
        :param args: keyword arguments that include default settings for the halo
        :param unique_tag: a random number with 16 decimal places that uniquely identifies each halo
        :param fixed_position: determiens whether halos can be moved around when aligning a realization with
        the rendering volume
        """

        self.lens_cosmo = lens_cosmo_instance
        self.mass = mass
        # x and y in arcsec
        self.x = x
        self.y = y
        self.r3d = r3d
        self.mdef = mdef
        self.z = z
        self.is_subhalo = sub_flag
        self._args = args
        self.unique_tag = unique_tag
        self._rescale_norm = 1.
        self.fixed_position = fixed_position
        self._rescaled_once = False

    def rescale_normalization(self, factor):
        """
        Sets the rescaling factor for the normalization (only can do this once)
        :param factor:
        :return:
        """
        if self._rescaled_once:
            pass
        else:
            self._rescaled_once = True
            self._rescale_norm = factor
            if hasattr(self, '_params_physical'):
                delattr(self, '_params_physical')
            if hasattr(self, '_kwargs_lenstronomy'):
                delattr(self, '_kwargs_lenstronomy')

    @property
    @abstractmethod
    def profile_args(self):
        """
        This routine computes properties of the halo required to specify it
        """
        ...

    @property
    def params_physical(self):
        raise Exception('this halo class does not have attribute params_physical')

    @property
    @abstractmethod
    def lenstronomy_ID(self):
        """
        Returns a list of profile names recognized by lenstronomy

        Example:
            a truncated NFW profile would return ['TNFW']
            a hybrid NFW and point mass profile would return ['NFW', 'POINT_MASS']

        """
        ...

    @property
    @abstractmethod
    def lenstronomy_params(self):
        """
        Returns a list of keyword arguments for the profile, must be the same length as lenstronomy_ID
        :return:
        """
        ...

    @property
    def z_infall(self):

        """
        Evaluate the infall redshift using a PDF generated by galacticus.
        Note: This routine is meaningless and therefore not used for LOS halos
        :return: the infall redshift of a halo assuming it is in a host halo at redshift self.z
        """

        if not hasattr(self, '_z_infall'):

            self._z_infall = self.lens_cosmo.z_accreted_from_zlens(self.mass, self.z)

        return self._z_infall

    @property
    def time_since_infall(self):
        """
        This routine calculates the time in Gyr since infall for subhalos using the infall redshift as predicted by
        Galacticus
        :return: the time since the subhalo was accreted onto thehost [Gyr]
        """
        if not self.is_subhalo:
            print("time since infall is a meaningless concept for field halos")
            return None
        if not hasattr(self, '_time_since_infall'):
            self._time_since_infall = self.lens_cosmo.cosmo.halo_age(self.z, self.z_infall)
        return self._time_since_infall

    @property
    def rperi_units_r200(self):
        """
        Returns the orbital pericenter of a subhalo in units of the host halo virial radius. This method
        uses output from the semi-analytic model Galacticus
        :return:
        """
        if not self.is_subhalo:
            print("Orbital pericenter is a meaningless concept for field halos. It is possible you assigned a tidal "
                  "truncation model that requires this information to field halos.")
            return None
        if not hasattr(self, '_rperi_units_r200'):
            self._rperi_units_r200 = _rperi_sampling(n_samples=1.0)
        return self._rperi_units_r200





