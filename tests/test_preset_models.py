from pyHalo.preset_models import WDM, CDM, SIDM, ULDM, preset_model_from_name
import numpy.testing as npt
import pytest
import numpy as np

class TestPresetModels(object):

    def test_CDM(self):
        
        realization_cdm = CDM(0.5, 1.5)
        npt.assert_equal(len(realization_cdm.rendering_classes), 3)
        realization_cdm = CDM(0.5, 1.5, c0=20.0, beta=0.9, zeta=-0.2)
        npt.assert_equal(len(realization_cdm.rendering_classes), 3)
        cdm = preset_model_from_name('CDM')
        realization_cdm = cdm(0.5, 1.5, log10c0=2.0, beta=0.9, zeta=-0.2)
        npt.assert_equal(len(realization_cdm.rendering_classes), 3)

    def test_WDM(self):

        realization_wdm = WDM(0.5, 1.5, 8.)
        npt.assert_equal(len(realization_wdm.rendering_classes), 3)
        kwargs_suppression_polynomial = {'c_power': -0.17, 'c_scale': 60.}
        kwargs_suppression_hyperbolic = {'a_mc': 1., 'b_mc': 1.}
        realization_wdm = WDM(0.5, 1.5, 8., suppression_model_field='polynomial',
                              suppression_model_sub='hyperbolic', kwargs_suppression_field=kwargs_suppression_polynomial,
                              kwargs_suppression_sub=kwargs_suppression_hyperbolic)
        npt.assert_equal(len(realization_wdm.rendering_classes), 3)

        realization_wdm = WDM(0.5, 1.5, 8., suppression_model_field='hyperbolic',
                              suppression_model_sub='polynomial',
                              kwargs_suppression_field=kwargs_suppression_hyperbolic,
                              kwargs_suppression_sub=kwargs_suppression_polynomial)
        npt.assert_equal(len(realization_wdm.rendering_classes), 3)

        wdm = preset_model_from_name('WDM')
        realization_wdm = wdm(0.5, 1.5, 8., suppression_model_field='hyperbolic',
                              suppression_model_sub='polynomial',
                              kwargs_suppression_field=kwargs_suppression_hyperbolic,
                              kwargs_suppression_sub=kwargs_suppression_polynomial)
        npt.assert_equal(len(realization_wdm.rendering_classes), 3)

    def test_SIDM(self):

        realization_SIDM = SIDM(0.5, 1.5, None, None, {}, {'x_core_halo': 0.05, 'log_slope_halo': 3.}, None, None,
                                None, None, None, sigma_sub=0., LOS_normalization=0.)
        npt.assert_equal(len(realization_SIDM.rendering_classes), 3)
        sidm = preset_model_from_name('SIDM')
        realization_SIDM = sidm(0.5, 1.5, None, None, {}, {'x_core_halo': 0.05, 'log_slope_halo': 3.}, None, None,
                                None, None, None, sigma_sub=0., LOS_normalization=0.)
        npt.assert_equal(len(realization_SIDM.rendering_classes), 3)

    def test_ULDM(self):

        log10_m_uldm=-22
        
        #test without fluctuations
        realization_ULDM = ULDM(0.5,1.5,log10_m_uldm,flucs=False)
        npt.assert_equal(len(realization_ULDM.rendering_classes), 3)
        uldm = preset_model_from_name('ULDM')
        realization_ULDM = uldm(0.5, 1.5,log10_m_uldm,flucs=False)
        npt.assert_equal(len(realization_ULDM.rendering_classes), 3)

        #test with fluctuations
        x_images = np.array([-0.347, -0.734, -1.096, 0.207])
        y_images = np.array([ 0.964,  0.649, -0.079, -0.148])
        flucs_args= {'x_images':x_images,'y_images':y_images,'aperture':0.25}

        realization_ULDM = ULDM(0.5,1.5,log10_m_uldm,flucs=True,shape='aperture',flucs_args=flucs_args)
        npt.assert_equal(len(realization_ULDM.rendering_classes), 3)
        uldm = preset_model_from_name('ULDM')
        realization_ULDM = uldm(0.5,1.5,log10_m_uldm,flucs=True,shape='aperture',flucs_args=flucs_args)
        npt.assert_equal(len(realization_ULDM.rendering_classes), 3)

if __name__ == '__main__':
     pytest.main()
