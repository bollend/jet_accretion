"""
Create the jet object
"""
import Cone

def create_jet(jet_type, pars, pars_walker):

    if jet_type=='stellar_jet_simple':

        jet = Cone.Stellar_jet_simple(pars_walker[pars['MODEL']['inclination']['id']],
                                      pars_walker[pars['MODEL']['jet_angle']['id']],
                                      pars_walker[pars['MODEL']['velocity_max']['id']],
                                      pars_walker[pars['MODEL']['velocity_edge']['id']],
                                      pars_walker[pars['MODEL']['power_density']['id']],
                                      jet_type,
                                      jet_tilt=pars_walker[pars['MODEL']['jet_tilt']['id']],
                                      double_tilt=pars['OTHER']['double_tilt'],
                                      jet_cavity_angle=pars_walker[pars['MODEL']['jet_cavity_angle']['id']]
                                      )

    elif jet_type=='stellar_jet':

        jet = Cone.Stellar_jet(pars_walker[pars['MODEL']['inclination']['id']],
                                      pars_walker[pars['MODEL']['jet_angle']['id']],
                                      pars_walker[pars['MODEL']['velocity_max']['id']],
                                      pars_walker[pars['MODEL']['velocity_edge']['id']],
                                      pars_walker[pars['MODEL']['exp_velocity']['id']],
                                      pars_walker[pars['MODEL']['power_density']['id']],
                                      jet_type,
                                      jet_tilt=pars_walker[pars['MODEL']['jet_tilt']['id']],
                                      double_tilt=pars['OTHER']['double_tilt'],
                                      jet_cavity_angle=pars_walker[pars['MODEL']['jet_cavity_angle']['id']]
                                      )

    elif jet_type=='x_wind':

        jet = Cone.X_wind(pars_walker[pars['MODEL']['inclination']['id']],
                                      pars_walker[pars['MODEL']['jet_angle']['id']],
                                      pars_walker[pars['MODEL']['velocity_max']['id']],
                                      pars_walker[pars['MODEL']['velocity_edge']['id']],
                                      pars_walker[pars['MODEL']['exp_velocity']['id']],
                                      pars_walker[pars['MODEL']['power_density_in']['id']],
                                      pars_walker[pars['MODEL']['power_density_out']['id']],
                                      jet_type,
                                      jet_tilt=pars_walker[pars['MODEL']['jet_tilt']['id']],
                                      double_tilt=pars['OTHER']['double_tilt'],
                                      jet_cavity_angle=pars_walker[pars['MODEL']['jet_cavity_angle']['id']],
                                      jet_angle_inner=pars_walker[pars['MODEL']['jet_angle_inner']['id']])

    elif jet_type=='x_wind_strict':

        jet = Cone.X_wind_strict(pars_walker[pars['MODEL']['inclination']['id']],
                                      pars_walker[pars['MODEL']['jet_angle']['id']],
                                      pars_walker[pars['MODEL']['velocity_max']['id']],
                                      pars_walker[pars['MODEL']['velocity_edge']['id']],
                                      pars_walker[pars['MODEL']['exp_velocity']['id']],
                                      pars_walker[pars['MODEL']['power_density']['id']],
                                      jet_type,
                                      jet_tilt=pars_walker[pars['MODEL']['jet_tilt']['id']],
                                      double_tilt=pars['OTHER']['double_tilt'],
                                      jet_cavity_angle=pars_walker[pars['MODEL']['jet_cavity_angle']['id']]
                                      )

    elif jet_type=='sdisk_wind':

        jet = Cone.Sdisk_wind(pars_walker[pars['MODEL']['inclination']['id']],
                                      pars_walker[pars['MODEL']['jet_angle']['id']],
                                      pars_walker[pars['MODEL']['velocity_max']['id']],
                                      pars_walker[pars['MODEL']['velocity_edge']['id']],
                                      pars_walker[pars['MODEL']['scaling_par']['id']],
                                      pars_walker[pars['MODEL']['power_density_in']['id']],
                                      pars_walker[pars['MODEL']['power_density_out']['id']],
                                      jet_type,
                                      jet_tilt=pars_walker[pars['MODEL']['jet_tilt']['id']],
                                      double_tilt=pars['OTHER']['double_tilt'],
                                      jet_cavity_angle=pars_walker[pars['MODEL']['jet_cavity_angle']['id']],
                                      jet_angle_inner=pars_walker[pars['MODEL']['jet_angle_inner']['id']]
                                      )

    elif jet_type=='sdisk_wind_strict':

        jet = Cone.Sdisk_wind_strict(pars_walker[pars['MODEL']['inclination']['id']],
                                      pars_walker[pars['MODEL']['jet_angle']['id']],
                                      pars_walker[pars['MODEL']['velocity_max']['id']],
                                      pars_walker[pars['MODEL']['velocity_edge']['id']],
                                      pars_walker[pars['MODEL']['scaling_par']['id']],
                                      pars_walker[pars['MODEL']['power_density']['id']],
                                      jet_type,
                                      jet_tilt=pars_walker[pars['MODEL']['jet_tilt']['id']],
                                      double_tilt=pars['OTHER']['double_tilt'],
                                      jet_cavity_angle=pars_walker[pars['MODEL']['jet_cavity_angle']['id']]
                                      )

    return jet
