"""
.. autoclass:: Thermochemistry
"""


import numpy as np


class Thermochemistry:
    """
    .. attribute:: model_name
    .. attribute:: num_elements
    .. attribute:: num_species
    .. attribute:: num_reactions
    .. attribute:: num_falloff
    .. attribute:: one_atm

        Returns 1 atm in SI units of pressure (Pa).

    .. attribute:: gas_constant
    .. attribute:: species_names
    .. attribute:: species_indices

    .. automethod:: get_specific_gas_constant
    .. automethod:: get_density
    .. automethod:: get_pressure
    .. automethod:: get_mix_molecular_weight
    .. automethod:: get_concentrations
    .. automethod:: get_mixture_specific_heat_cp_mass
    .. automethod:: get_mixture_specific_heat_cv_mass
    .. automethod:: get_mixture_enthalpy_mass
    .. automethod:: get_mixture_internal_energy_mass
    .. automethod:: get_species_viscosities
    .. automethod:: get_mixture_viscosity_mixavg
    .. automethod:: get_species_thermal_conductivities
    .. automethod:: get_mixture_thermal_conductivity_mixavg
    .. automethod:: get_species_binary_mass_diffusivities
    .. automethod:: get_species_mass_diffusivities_mixavg
    .. automethod:: get_species_specific_heats_r
    .. automethod:: get_species_enthalpies_rt
    .. automethod:: get_species_entropies_r
    .. automethod:: get_species_gibbs_rt
    .. automethod:: get_equilibrium_constants
    .. automethod:: get_temperature
    .. automethod:: __init__
    """

    def __init__(self, usr_np=np):
        """Initialize thermochemistry object for a mechanism.

        Parameters
        ----------
        usr_np
            :mod:`numpy`-like namespace providing at least the following functions,
            for any array ``X`` of the bulk array type:

            - ``usr_np.log(X)`` (like :data:`numpy.log`)
            - ``usr_np.log10(X)`` (like :data:`numpy.log10`)
            - ``usr_np.exp(X)`` (like :data:`numpy.exp`)
            - ``usr_np.where(X > 0, X_yes, X_no)`` (like :func:`numpy.where`)
            - ``usr_np.linalg.norm(X, np.inf)`` (like :func:`numpy.linalg.norm`)

            where the "bulk array type" is a type that offers arithmetic analogous
            to :class:`numpy.ndarray` and is used to hold all types of (potentialy
            volumetric) "bulk data", such as temperature, pressure, mass fractions,
            etc. This parameter defaults to *actual numpy*, so it can be ignored
            unless it is needed by the user (e.g. for purposes of
            GPU processing or automatic differentiation).

        """

        self.usr_np = usr_np
        self.model_name = 'mechs/uiuc_mod.yaml'
        self.num_elements = 4
        self.num_species = 7
        self.num_reactions = 3
        self.num_falloff = 0

        self.one_atm = 101325.0
        self.gas_constant = 8314.46261815324
        self.big_number = 1.0e300

        self.species_names = ['C2H4', 'O2', 'CO2', 'CO', 'H2O', 'H2', 'N2']
        self.species_indices = {'C2H4': 0, 'O2': 1, 'CO2': 2, 'CO': 3, 'H2O': 4, 'H2': 5, 'N2': 6}

        self.wts = np.array([28.054, 31.998, 44.009, 28.009999999999998, 18.015, 2.016, 28.014])
        self.iwts = 1/self.wts

    def _pyro_zeros_like(self, argument):
        # FIXME: This is imperfect, as a NaN will stay a NaN.
        return 0 * argument

    def _pyro_make_array(self, res_list):
        """This works around (e.g.) numpy.exp not working with object
        arrays of numpy scalars. It defaults to making object arrays, however
        if an array consists of all scalars, it makes a "plain old"
        :class:`numpy.ndarray`.

        See ``this numpy bug <https://github.com/numpy/numpy/issues/18004>`__
        for more context.
        """

        from numbers import Number
        all_numbers = all(isinstance(e, Number) for e in res_list)

        dtype = np.float64 if all_numbers else object
        result = np.empty((len(res_list),), dtype=dtype)

        # 'result[:] = res_list' may look tempting, however:
        # https://github.com/numpy/numpy/issues/16564
        for idx in range(len(res_list)):
            result[idx] = res_list[idx]

        return result

    def _pyro_norm(self, argument, normord):
        """This works around numpy.linalg norm not working with scalars.

        If the argument is a regular ole number, it uses :func:`numpy.abs`,
        otherwise it uses ``usr_np.linalg.norm``.
        """
        # Wrap norm for scalars

        from numbers import Number

        if isinstance(argument, Number):
            return np.abs(argument)
        return self.usr_np.linalg.norm(argument, normord)

    def species_name(self, species_index):
        return self.species_name[species_index]

    def species_index(self, species_name):
        return self.species_indices[species_name]

    def get_specific_gas_constant(self, mass_fractions):
        return self.gas_constant * (
            + self.iwts[0]*mass_fractions[0]
            + self.iwts[1]*mass_fractions[1]
            + self.iwts[2]*mass_fractions[2]
            + self.iwts[3]*mass_fractions[3]
            + self.iwts[4]*mass_fractions[4]
            + self.iwts[5]*mass_fractions[5]
            + self.iwts[6]*mass_fractions[6]
        )

    def get_density(self, p, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        rt = self.gas_constant * temperature
        return p * mmw / rt

    def get_pressure(self, rho, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        rt = self.gas_constant * temperature
        return rho * rt / mmw

    def get_mix_molecular_weight(self, mass_fractions):
        return 1/(
            + self.iwts[0]*mass_fractions[0]
            + self.iwts[1]*mass_fractions[1]
            + self.iwts[2]*mass_fractions[2]
            + self.iwts[3]*mass_fractions[3]
            + self.iwts[4]*mass_fractions[4]
            + self.iwts[5]*mass_fractions[5]
            + self.iwts[6]*mass_fractions[6]
        )

    def get_concentrations(self, rho, mass_fractions):
        return self._pyro_make_array([
            self.iwts[0] * rho * mass_fractions[0],
            self.iwts[1] * rho * mass_fractions[1],
            self.iwts[2] * rho * mass_fractions[2],
            self.iwts[3] * rho * mass_fractions[3],
            self.iwts[4] * rho * mass_fractions[4],
            self.iwts[5] * rho * mass_fractions[5],
            self.iwts[6] * rho * mass_fractions[6],
        ])

    def get_mole_fractions(self, mix_mol_weight, mass_fractions):
        return self._pyro_make_array([
            self.iwts[0] * mass_fractions[0] * mix_mol_weight,
            self.iwts[1] * mass_fractions[1] * mix_mol_weight,
            self.iwts[2] * mass_fractions[2] * mix_mol_weight,
            self.iwts[3] * mass_fractions[3] * mix_mol_weight,
            self.iwts[4] * mass_fractions[4] * mix_mol_weight,
            self.iwts[5] * mass_fractions[5] * mix_mol_weight,
            self.iwts[6] * mass_fractions[6] * mix_mol_weight,
        ])

    def get_mass_average_property(self, mass_fractions, spec_property):
        return sum([mass_fractions[i] * spec_property[i] * self.iwts[i]
                    for i in range(self.num_species)])

    def get_mixture_specific_heat_cp_mass(self, temperature, mass_fractions):
        cp0_r = self.get_species_specific_heats_r(temperature)
        cpmix = self.get_mass_average_property(mass_fractions, cp0_r)
        return self.gas_constant * cpmix

    def get_mixture_specific_heat_cv_mass(self, temperature, mass_fractions):
        cp0_r = self.get_species_specific_heats_r(temperature) - 1.0
        cpmix = self.get_mass_average_property(mass_fractions, cp0_r)
        return self.gas_constant * cpmix

    def get_mixture_enthalpy_mass(self, temperature, mass_fractions):
        h0_rt = self.get_species_enthalpies_rt(temperature)
        hmix = self.get_mass_average_property(mass_fractions, h0_rt)
        return self.gas_constant * temperature * hmix

    def get_mixture_internal_energy_mass(self, temperature, mass_fractions):
        e0_rt = self.get_species_enthalpies_rt(temperature) - 1.0
        emix = self.get_mass_average_property(mass_fractions, e0_rt)
        return self.gas_constant * temperature * emix

    def get_species_viscosities(self, temperature):
        return self._pyro_make_array([
                self.usr_np.sqrt(temperature)*(0.0005643720009284571 + -0.0008536047562446639*self.usr_np.log(temperature) + 0.00034065536407341777*self.usr_np.log(temperature)**2 + -4.2648038931472564e-05*self.usr_np.log(temperature)**3 + 1.7934368018097565e-06*self.usr_np.log(temperature)**4)**2,
                self.usr_np.sqrt(temperature)*(-0.006186428071784459 + 0.003618824512265024*self.usr_np.log(temperature) + -0.0006861983404519474*self.usr_np.log(temperature)**2 + 5.916012709644184e-05*self.usr_np.log(temperature)**3 + -1.9049771784433379e-06*self.usr_np.log(temperature)**4)**2,
                self.usr_np.sqrt(temperature)*(-0.0027060422317322606 + 0.00099608913488712*self.usr_np.log(temperature) + -3.245759288262185e-05*self.usr_np.log(temperature)**2 + -9.15207993467064e-06*self.usr_np.log(temperature)**3 + 6.716178143243158e-07*self.usr_np.log(temperature)**4)**2,
                self.usr_np.sqrt(temperature)*(-0.00521608145079593 + 0.003111199674194881*self.usr_np.log(temperature) + -0.0005939529792207282*self.usr_np.log(temperature)**2 + 5.162313384558195e-05*self.usr_np.log(temperature)**3 + -1.6749040550313004e-06*self.usr_np.log(temperature)**4)**2,
                self.usr_np.sqrt(temperature)*(0.009495196334954333 + -0.004974400618445216*self.usr_np.log(temperature) + 0.0009719845681945996*self.usr_np.log(temperature)**2 + -7.63468726049601e-05*self.usr_np.log(temperature)**3 + 2.0741201775470118e-06*self.usr_np.log(temperature)**4)**2,
                self.usr_np.sqrt(temperature)*(-0.00032862351581974033 + 0.00047402944328694686*self.usr_np.log(temperature) + -8.852339013643083e-05*self.usr_np.log(temperature)**2 + 8.188000383424178e-06*self.usr_np.log(temperature)**3 + -2.775116846661831e-07*self.usr_np.log(temperature)**4)**2,
                self.usr_np.sqrt(temperature)*(-0.005232503458857859 + 0.003124981120246175*self.usr_np.log(temperature) + -0.0005968857242794555*self.usr_np.log(temperature)**2 + 5.190839695047409e-05*self.usr_np.log(temperature)**3 + -1.6850989392761648e-06*self.usr_np.log(temperature)**4)**2,
                ])

    def get_mixture_viscosity_mixavg(self, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        mole_fracs = self.get_mole_fractions(mmw, mass_fractions)
        viscosities = self.get_species_viscosities(temperature)
        mix_rule_f = self._pyro_make_array([
            (mole_fracs[0]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[0])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fracs[1]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[1])*self.usr_np.sqrt(1.1405860126898126)))**2) / self.usr_np.sqrt(15.013938371148196) + (mole_fracs[2]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[2])*self.usr_np.sqrt(1.5687246025522208)))**2) / self.usr_np.sqrt(13.099684155513645) + (mole_fracs[3]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[3])*self.usr_np.sqrt(0.9984315962073145)))**2) / self.usr_np.sqrt(16.012566940378434) + (mole_fracs[4]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[4])*self.usr_np.sqrt(0.642154416482498)))**2) / self.usr_np.sqrt(20.458062725506522) + (mole_fracs[5]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[5])*self.usr_np.sqrt(0.07186141013759179)))**2) / self.usr_np.sqrt(119.32539682539682) + (mole_fracs[6]*(1 + self.usr_np.sqrt((viscosities[0] / viscosities[6])*self.usr_np.sqrt(0.998574178370286)))**2) / self.usr_np.sqrt(16.01142285999857),
            (mole_fracs[0]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[0])*self.usr_np.sqrt(0.8767422963935245)))**2) / self.usr_np.sqrt(17.1246881015185) + (mole_fracs[1]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[1])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fracs[2]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[2])*self.usr_np.sqrt(1.3753672104506531)))**2) / self.usr_np.sqrt(13.816628416914721) + (mole_fracs[3]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[3])*self.usr_np.sqrt(0.875367210450653)))**2) / self.usr_np.sqrt(17.139021777936453) + (mole_fracs[4]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[4])*self.usr_np.sqrt(0.5630039377461091)))**2) / self.usr_np.sqrt(22.209492089925064) + (mole_fracs[5]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[5])*self.usr_np.sqrt(0.06300393774610913)))**2) / self.usr_np.sqrt(134.97619047619048) + (mole_fracs[6]*(1 + self.usr_np.sqrt((viscosities[1] / viscosities[6])*self.usr_np.sqrt(0.8754922182636414)))**2) / self.usr_np.sqrt(17.137716855857786),
            (mole_fracs[0]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[0])*self.usr_np.sqrt(0.6374605194392056)))**2) / self.usr_np.sqrt(20.549796820417768) + (mole_fracs[1]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[1])*self.usr_np.sqrt(0.7270785521143402)))**2) / self.usr_np.sqrt(19.002937683605225) + (mole_fracs[2]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[2])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fracs[3]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[3])*self.usr_np.sqrt(0.6364607239428298)))**2) / self.usr_np.sqrt(20.569510888968225) + (mole_fracs[4]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[4])*self.usr_np.sqrt(0.4093480878911132)))**2) / self.usr_np.sqrt(27.543269497640853) + (mole_fracs[5]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[5])*self.usr_np.sqrt(0.04580881183394306)))**2) / self.usr_np.sqrt(182.63888888888889) + (mole_fracs[6]*(1 + self.usr_np.sqrt((viscosities[2] / viscosities[6])*self.usr_np.sqrt(0.6365516144425004)))**2) / self.usr_np.sqrt(20.567716141929036),
            (mole_fracs[0]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[0])*self.usr_np.sqrt(1.0015708675473045)))**2) / self.usr_np.sqrt(15.987452769658516) + (mole_fracs[1]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[1])*self.usr_np.sqrt(1.1423777222420566)))**2) / self.usr_np.sqrt(15.002937683605225) + (mole_fracs[2]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[2])*self.usr_np.sqrt(1.5711888611210283)))**2) / self.usr_np.sqrt(13.091685791542638) + (mole_fracs[3]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[3])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fracs[4]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[4])*self.usr_np.sqrt(0.6431631560157087)))**2) / self.usr_np.sqrt(20.43852345267832) + (mole_fracs[5]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[5])*self.usr_np.sqrt(0.07197429489468048)))**2) / self.usr_np.sqrt(119.15079365079364) + (mole_fracs[6]*(1 + self.usr_np.sqrt((viscosities[3] / viscosities[6])*self.usr_np.sqrt(1.000142806140664)))**2) / self.usr_np.sqrt(15.998857714000142),
            (mole_fracs[0]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[0])*self.usr_np.sqrt(1.557257840688315)))**2) / self.usr_np.sqrt(13.137235331859983) + (mole_fracs[1]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[1])*self.usr_np.sqrt(1.7761865112406328)))**2) / self.usr_np.sqrt(12.504031501968873) + (mole_fracs[2]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[2])*self.usr_np.sqrt(2.4429086872051067)))**2) / self.usr_np.sqrt(11.274784703128905) + (mole_fracs[3]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[3])*self.usr_np.sqrt(1.5548154315847902)))**2) / self.usr_np.sqrt(13.14530524812567) + (mole_fracs[4]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[4])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fracs[5]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[5])*self.usr_np.sqrt(0.11190674437968359)))**2) / self.usr_np.sqrt(79.48809523809524) + (mole_fracs[6]*(1 + self.usr_np.sqrt((viscosities[4] / viscosities[6])*self.usr_np.sqrt(1.55503746877602)))**2) / self.usr_np.sqrt(13.144570571856928),
            (mole_fracs[0]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[0])*self.usr_np.sqrt(13.915674603174603)))**2) / self.usr_np.sqrt(8.574891281100735) + (mole_fracs[1]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[1])*self.usr_np.sqrt(15.87202380952381)))**2) / self.usr_np.sqrt(8.504031501968873) + (mole_fracs[2]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[2])*self.usr_np.sqrt(21.82986111111111)))**2) / self.usr_np.sqrt(8.366470494671544) + (mole_fracs[3]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[3])*self.usr_np.sqrt(13.893849206349206)))**2) / self.usr_np.sqrt(8.575794359157443) + (mole_fracs[4]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[4])*self.usr_np.sqrt(8.936011904761905)))**2) / self.usr_np.sqrt(8.895253955037468) + (mole_fracs[5]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[5])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0) + (mole_fracs[6]*(1 + self.usr_np.sqrt((viscosities[5] / viscosities[6])*self.usr_np.sqrt(13.895833333333332)))**2) / self.usr_np.sqrt(8.575712143928037),
            (mole_fracs[0]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[0])*self.usr_np.sqrt(1.0014278574998214)))**2) / self.usr_np.sqrt(15.988593426962288) + (mole_fracs[1]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[1])*self.usr_np.sqrt(1.1422146069822232)))**2) / self.usr_np.sqrt(15.00393774610913) + (mole_fracs[2]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[2])*self.usr_np.sqrt(1.5709645177411296)))**2) / self.usr_np.sqrt(13.092412915540002) + (mole_fracs[3]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[3])*self.usr_np.sqrt(0.9998572142500178)))**2) / self.usr_np.sqrt(16.00114244912531) + (mole_fracs[4]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[4])*self.usr_np.sqrt(0.6430713214821161)))**2) / self.usr_np.sqrt(20.440299750208162) + (mole_fracs[5]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[5])*self.usr_np.sqrt(0.0719640179910045)))**2) / self.usr_np.sqrt(119.16666666666666) + (mole_fracs[6]*(1 + self.usr_np.sqrt((viscosities[6] / viscosities[6])*self.usr_np.sqrt(1.0)))**2) / self.usr_np.sqrt(16.0),
            ])
        return sum(mole_fracs*viscosities/mix_rule_f)

    def get_species_thermal_conductivities(self, temperature):
        return self._pyro_make_array([
                self.usr_np.sqrt(temperature)*(0.18385271624866842 + -0.09812050654864941*self.usr_np.log(temperature) + 0.01871067923066944*self.usr_np.log(temperature)**2 + -0.0014894434590280682*self.usr_np.log(temperature)**3 + 4.220932619231024e-05*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*(0.10689552101630528 + -0.06376711343770658*self.usr_np.log(temperature) + 0.014217756178712414*self.usr_np.log(temperature)**2 + -0.0013908411531386269*self.usr_np.log(temperature)**3 + 5.09174438888175e-05*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*(0.0871088646655998 + -0.05136577647286977*self.usr_np.log(temperature) + 0.011079892418410776*self.usr_np.log(temperature)**2 + -0.0010291064627698154*self.usr_np.log(temperature)**3 + 3.531690263236322e-05*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*(0.04060029461887554 + -0.020985470902516137*self.usr_np.log(temperature) + 0.00400137335308852*self.usr_np.log(temperature)**2 + -0.00032019801523515564*self.usr_np.log(temperature)**3 + 9.270576537591015e-06*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*(-0.40448952246714476 + 0.25166528584206543*self.usr_np.log(temperature) + -0.05823800028133604*self.usr_np.log(temperature)**2 + 0.005930903658833211*self.usr_np.log(temperature)**3 + -0.00022233754286936298*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*(-0.9677034329303261 + 0.5744337660316912*self.usr_np.log(temperature) + -0.12573711513496563*self.usr_np.log(temperature)**2 + 0.012123569772975705*self.usr_np.log(temperature)**3 + -0.0004317820757905789*self.usr_np.log(temperature)**4),
                self.usr_np.sqrt(temperature)*(0.0026129214099176023 + 0.001593238644698187*self.usr_np.log(temperature) + -0.0009842775277398488*self.usr_np.log(temperature)**2 + 0.00016507154037197767*self.usr_np.log(temperature)**3 + -8.29731531537306e-06*self.usr_np.log(temperature)**4),
                ])

    def get_mixture_thermal_conductivity_mixavg(self, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        mole_fracs = self.get_mole_fractions(mmw, mass_fractions)
        conductivities = self.get_species_thermal_conductivities(temperature)
        return 0.5*(sum(mole_fracs*conductivities)
            + 1/sum(mole_fracs/conductivities))

    def get_species_binary_mass_diffusivities(self, temperature):
        return self._pyro_make_array([
                0.0025140251674159882 + -0.0017759952625790237*self.usr_np.log(temperature) + 0.00044874581158390963*self.usr_np.log(temperature)**2 + -4.6325289302623295e-05*self.usr_np.log(temperature)**3 + 1.7399023936056533e-06*self.usr_np.log(temperature)**4,
                -0.0017328702787483422 + 0.0006970694562140113*self.usr_np.log(temperature) + -7.498493379341564e-05*self.usr_np.log(temperature)**2 + 2.3790313072646865e-06*self.usr_np.log(temperature)**3 + 6.258149854115358e-08*self.usr_np.log(temperature)**4,
                0.0016279660308428918 + -0.001251861155981191*self.usr_np.log(temperature) + 0.00033501937503295915*self.usr_np.log(temperature)**2 + -3.560522825076669e-05*self.usr_np.log(temperature)**3 + 1.365815818767913e-06*self.usr_np.log(temperature)**4,
                -0.0019075474076543904 + 0.0008156938750998444*self.usr_np.log(temperature) + -0.00010330573524890039*self.usr_np.log(temperature)**2 + 5.235151668307418e-06*self.usr_np.log(temperature)**3 + -4.2437646766789564e-08*self.usr_np.log(temperature)**4,
                0.011422240725233268 + -0.006973463253996788*self.usr_np.log(temperature) + 0.001564784800280144*self.usr_np.log(temperature)**2 + -0.00015019828712470825*self.usr_np.log(temperature)**3 + 5.320779247502136e-06*self.usr_np.log(temperature)**4,
                -0.009051662147541151 + 0.004761002619177032*self.usr_np.log(temperature) + -0.0008507988153589192*self.usr_np.log(temperature)**2 + 7.018609346583454e-05*self.usr_np.log(temperature)**3 + -2.118223235585378e-06*self.usr_np.log(temperature)**4,
                -0.0019364269556271293 + 0.0008309848230403403*self.usr_np.log(temperature) + -0.00010613902008251093*self.usr_np.log(temperature)**2 + 5.474741526097051e-06*self.usr_np.log(temperature)**3 + -4.991875444829921e-08*self.usr_np.log(temperature)**4,
                -0.0017328702787483422 + 0.0006970694562140113*self.usr_np.log(temperature) + -7.498493379341564e-05*self.usr_np.log(temperature)**2 + 2.3790313072646865e-06*self.usr_np.log(temperature)**3 + 6.258149854115358e-08*self.usr_np.log(temperature)**4,
                -0.0031272899371028937 + 0.0016332321885213815*self.usr_np.log(temperature) + -0.0002902324473161449*self.usr_np.log(temperature)**2 + 2.3795154190814967e-05*self.usr_np.log(temperature)**3 + -7.13575445903519e-07*self.usr_np.log(temperature)**4,
                -0.0019538762606278884 + 0.0008581597928110169*self.usr_np.log(temperature) + -0.00011552373820505783*self.usr_np.log(temperature)**2 + 6.5880314073473246e-06*self.usr_np.log(temperature)**3 + -9.570226239146685e-08*self.usr_np.log(temperature)**4,
                -0.003009672869051961 + 0.0015847679011310326*self.usr_np.log(temperature) + -0.0002834159166131279*self.usr_np.log(temperature)**2 + 2.3400497455310064e-05*self.usr_np.log(temperature)**3 + -7.068324598250149e-07*self.usr_np.log(temperature)**4,
                0.004416855007526046 + -0.0031775743855889087*self.usr_np.log(temperature) + 0.0008133764546304064*self.usr_np.log(temperature)**2 + -8.454231010250692e-05*self.usr_np.log(temperature)**3 + 3.1913607821509437e-06*self.usr_np.log(temperature)**4,
                -0.008078545001733557 + 0.004640788076964306*self.usr_np.log(temperature) + -0.0008759648030327076*self.usr_np.log(temperature)**2 + 7.70576987840301e-05*self.usr_np.log(temperature)**3 + -2.4732690890930056e-06*self.usr_np.log(temperature)**4,
                -0.00303005347627657 + 0.0015963119455954615*self.usr_np.log(temperature) + -0.000285582712380885*self.usr_np.log(temperature)**2 + 2.3589003022593766e-05*self.usr_np.log(temperature)**3 + -7.128139183159607e-07*self.usr_np.log(temperature)**4,
                0.0016279660308428918 + -0.001251861155981191*self.usr_np.log(temperature) + 0.00033501937503295915*self.usr_np.log(temperature)**2 + -3.560522825076669e-05*self.usr_np.log(temperature)**3 + 1.365815818767913e-06*self.usr_np.log(temperature)**4,
                -0.0019538762606278884 + 0.0008581597928110169*self.usr_np.log(temperature) + -0.00011552373820505783*self.usr_np.log(temperature)**2 + 6.5880314073473246e-06*self.usr_np.log(temperature)**3 + -9.570226239146685e-08*self.usr_np.log(temperature)**4,
                0.0008536776054451306 + -0.0007825552849370942*self.usr_np.log(temperature) + 0.0002308878438631842*self.usr_np.log(temperature)**2 + -2.5636905937343826e-05*self.usr_np.log(temperature)**3 + 1.0133819725254011e-06*self.usr_np.log(temperature)**4,
                -0.0020933310545413466 + 0.0009551433118501238*self.usr_np.log(temperature) + -0.00013894323924026357*self.usr_np.log(temperature)**2 + 8.974603118125933e-06*self.usr_np.log(temperature)**3 + -1.841695259121606e-07*self.usr_np.log(temperature)**4,
                0.011799896789229356 + -0.0071765471397073535*self.usr_np.log(temperature) + 0.001604679853871194*self.usr_np.log(temperature)**2 + -0.0001536151265013595*self.usr_np.log(temperature)**3 + 5.428853321346705e-06*self.usr_np.log(temperature)**4,
                -0.009335286151716984 + 0.004977408057966928*self.usr_np.log(temperature) + -0.0009000585925404341*self.usr_np.log(temperature)**2 + 7.519297228980869e-05*self.usr_np.log(temperature)**3 + -2.300042056791817e-06*self.usr_np.log(temperature)**4,
                -0.002119434380751724 + 0.0009691103714403257*self.usr_np.log(temperature) + -0.0001415437410210921*self.usr_np.log(temperature)**2 + 9.196204490091708e-06*self.usr_np.log(temperature)**3 + -1.911397837450635e-07*self.usr_np.log(temperature)**4,
                -0.0019075474076543904 + 0.0008156938750998444*self.usr_np.log(temperature) + -0.00010330573524890039*self.usr_np.log(temperature)**2 + 5.235151668307418e-06*self.usr_np.log(temperature)**3 + -4.2437646766789564e-08*self.usr_np.log(temperature)**4,
                -0.003009672869051961 + 0.0015847679011310326*self.usr_np.log(temperature) + -0.0002834159166131279*self.usr_np.log(temperature)**2 + 2.3400497455310064e-05*self.usr_np.log(temperature)**3 + -7.068324598250149e-07*self.usr_np.log(temperature)**4,
                -0.0020933310545413466 + 0.0009551433118501238*self.usr_np.log(temperature) + -0.00013894323924026357*self.usr_np.log(temperature)**2 + 8.974603118125933e-06*self.usr_np.log(temperature)**3 + -1.841695259121606e-07*self.usr_np.log(temperature)**4,
                -0.0029131057507519637 + 0.0015474779150422674*self.usr_np.log(temperature) + -0.0002788854867879387*self.usr_np.log(temperature)**2 + 2.3216200335934344e-05*self.usr_np.log(temperature)**3 + -7.074354736540356e-07*self.usr_np.log(temperature)**4,
                0.0034801095778207868 + -0.002616545671615215*self.usr_np.log(temperature) + 0.0006901368404514836*self.usr_np.log(temperature)**2 + -7.282721342074532e-05*self.usr_np.log(temperature)**3 + 2.779507883753769e-06*self.usr_np.log(temperature)**4,
                -0.007285927271240993 + 0.004223298411073273*self.usr_np.log(temperature) + -0.0007987477271398378*self.usr_np.log(temperature)**2 + 7.054617033839934e-05*self.usr_np.log(temperature)**3 + -2.2704437485799014e-06*self.usr_np.log(temperature)**4,
                -0.0029348181666659554 + 0.0015599234311282119*self.usr_np.log(temperature) + -0.00028128084862643775*self.usr_np.log(temperature)**2 + 2.3428922377910606e-05*self.usr_np.log(temperature)**3 + -7.143582738260394e-07*self.usr_np.log(temperature)**4,
                0.011422240725233268 + -0.006973463253996788*self.usr_np.log(temperature) + 0.001564784800280144*self.usr_np.log(temperature)**2 + -0.00015019828712470825*self.usr_np.log(temperature)**3 + 5.320779247502136e-06*self.usr_np.log(temperature)**4,
                0.004416855007526046 + -0.0031775743855889087*self.usr_np.log(temperature) + 0.0008133764546304064*self.usr_np.log(temperature)**2 + -8.454231010250692e-05*self.usr_np.log(temperature)**3 + 3.1913607821509437e-06*self.usr_np.log(temperature)**4,
                0.011799896789229356 + -0.0071765471397073535*self.usr_np.log(temperature) + 0.001604679853871194*self.usr_np.log(temperature)**2 + -0.0001536151265013595*self.usr_np.log(temperature)**3 + 5.428853321346705e-06*self.usr_np.log(temperature)**4,
                0.0034801095778207868 + -0.002616545671615215*self.usr_np.log(temperature) + 0.0006901368404514836*self.usr_np.log(temperature)**2 + -7.282721342074532e-05*self.usr_np.log(temperature)**3 + 2.779507883753769e-06*self.usr_np.log(temperature)**4,
                0.008153691915849715 + -0.0039502872622226674*self.usr_np.log(temperature) + 0.0006415133652066968*self.usr_np.log(temperature)**2 + -3.4901825620169965e-05*self.usr_np.log(temperature)**3 + 3.21842467406017e-07*self.usr_np.log(temperature)**4,
                -0.009701326278443632 + 0.004014323899384639*self.usr_np.log(temperature) + -0.00046791095879321343*self.usr_np.log(temperature)**2 + 1.938085265871765e-05*self.usr_np.log(temperature)**3 + 9.241023547824361e-08*self.usr_np.log(temperature)**4,
                0.0032788605181502195 + -0.0025054784514831137*self.usr_np.log(temperature) + 0.0006678165711074384*self.usr_np.log(temperature)**2 + -7.083588486055441e-05*self.usr_np.log(temperature)**3 + 2.7134926982455882e-06*self.usr_np.log(temperature)**4,
                -0.009051662147541151 + 0.004761002619177032*self.usr_np.log(temperature) + -0.0008507988153589192*self.usr_np.log(temperature)**2 + 7.018609346583454e-05*self.usr_np.log(temperature)**3 + -2.118223235585378e-06*self.usr_np.log(temperature)**4,
                -0.008078545001733557 + 0.004640788076964306*self.usr_np.log(temperature) + -0.0008759648030327076*self.usr_np.log(temperature)**2 + 7.70576987840301e-05*self.usr_np.log(temperature)**3 + -2.4732690890930056e-06*self.usr_np.log(temperature)**4,
                -0.009335286151716984 + 0.004977408057966928*self.usr_np.log(temperature) + -0.0009000585925404341*self.usr_np.log(temperature)**2 + 7.519297228980869e-05*self.usr_np.log(temperature)**3 + -2.300042056791817e-06*self.usr_np.log(temperature)**4,
                -0.007285927271240993 + 0.004223298411073273*self.usr_np.log(temperature) + -0.0007987477271398378*self.usr_np.log(temperature)**2 + 7.054617033839934e-05*self.usr_np.log(temperature)**3 + -2.2704437485799014e-06*self.usr_np.log(temperature)**4,
                -0.009701326278443632 + 0.004014323899384639*self.usr_np.log(temperature) + -0.00046791095879321343*self.usr_np.log(temperature)**2 + 1.938085265871765e-05*self.usr_np.log(temperature)**3 + 9.241023547824361e-08*self.usr_np.log(temperature)**4,
                -0.006865233575679116 + 0.00452798836275456*self.usr_np.log(temperature) + -0.0008654211810305249*self.usr_np.log(temperature)**2 + 7.992970847168102e-05*self.usr_np.log(temperature)**3 + -2.6377961782905754e-06*self.usr_np.log(temperature)**4,
                -0.007323280548457206 + 0.004247486239670019*self.usr_np.log(temperature) + -0.000803359454232897*self.usr_np.log(temperature)**2 + 7.096837772986983e-05*self.usr_np.log(temperature)**3 + -2.2842690699017418e-06*self.usr_np.log(temperature)**4,
                -0.0019364269556271293 + 0.0008309848230403403*self.usr_np.log(temperature) + -0.00010613902008251093*self.usr_np.log(temperature)**2 + 5.474741526097051e-06*self.usr_np.log(temperature)**3 + -4.991875444829921e-08*self.usr_np.log(temperature)**4,
                -0.00303005347627657 + 0.0015963119455954615*self.usr_np.log(temperature) + -0.000285582712380885*self.usr_np.log(temperature)**2 + 2.3589003022593766e-05*self.usr_np.log(temperature)**3 + -7.128139183159607e-07*self.usr_np.log(temperature)**4,
                -0.002119434380751724 + 0.0009691103714403257*self.usr_np.log(temperature) + -0.0001415437410210921*self.usr_np.log(temperature)**2 + 9.196204490091708e-06*self.usr_np.log(temperature)**3 + -1.911397837450635e-07*self.usr_np.log(temperature)**4,
                -0.0029348181666659554 + 0.0015599234311282119*self.usr_np.log(temperature) + -0.00028128084862643775*self.usr_np.log(temperature)**2 + 2.3428922377910606e-05*self.usr_np.log(temperature)**3 + -7.143582738260394e-07*self.usr_np.log(temperature)**4,
                0.0032788605181502195 + -0.0025054784514831137*self.usr_np.log(temperature) + 0.0006678165711074384*self.usr_np.log(temperature)**2 + -7.083588486055441e-05*self.usr_np.log(temperature)**3 + 2.7134926982455882e-06*self.usr_np.log(temperature)**4,
                -0.007323280548457206 + 0.004247486239670019*self.usr_np.log(temperature) + -0.000803359454232897*self.usr_np.log(temperature)**2 + 7.096837772986983e-05*self.usr_np.log(temperature)**3 + -2.2842690699017418e-06*self.usr_np.log(temperature)**4,
                -0.002956738522928091 + 0.001572491741903353*self.usr_np.log(temperature) + -0.00028369990578656267*self.usr_np.log(temperature)**2 + 2.3643775746067122e-05*self.usr_np.log(temperature)**3 + -7.213510385692085e-07*self.usr_np.log(temperature)**4,
                ]).reshape((self.num_species, self.num_species))

    def get_species_mass_diffusivities_mixavg(self, pressure, temperature,
                                              mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        mole_fracs = self.get_mole_fractions(mmw, mass_fractions)
        bdiff_ij = self.get_species_binary_mass_diffusivities(temperature)

        x_sum = self._pyro_make_array([
            mole_fracs[0] / bdiff_ij[0, 0] + mole_fracs[1] / bdiff_ij[1, 0] + mole_fracs[2] / bdiff_ij[2, 0] + mole_fracs[3] / bdiff_ij[3, 0] + mole_fracs[4] / bdiff_ij[4, 0] + mole_fracs[5] / bdiff_ij[5, 0] + mole_fracs[6] / bdiff_ij[6, 0],
            mole_fracs[0] / bdiff_ij[0, 1] + mole_fracs[1] / bdiff_ij[1, 1] + mole_fracs[2] / bdiff_ij[2, 1] + mole_fracs[3] / bdiff_ij[3, 1] + mole_fracs[4] / bdiff_ij[4, 1] + mole_fracs[5] / bdiff_ij[5, 1] + mole_fracs[6] / bdiff_ij[6, 1],
            mole_fracs[0] / bdiff_ij[0, 2] + mole_fracs[1] / bdiff_ij[1, 2] + mole_fracs[2] / bdiff_ij[2, 2] + mole_fracs[3] / bdiff_ij[3, 2] + mole_fracs[4] / bdiff_ij[4, 2] + mole_fracs[5] / bdiff_ij[5, 2] + mole_fracs[6] / bdiff_ij[6, 2],
            mole_fracs[0] / bdiff_ij[0, 3] + mole_fracs[1] / bdiff_ij[1, 3] + mole_fracs[2] / bdiff_ij[2, 3] + mole_fracs[3] / bdiff_ij[3, 3] + mole_fracs[4] / bdiff_ij[4, 3] + mole_fracs[5] / bdiff_ij[5, 3] + mole_fracs[6] / bdiff_ij[6, 3],
            mole_fracs[0] / bdiff_ij[0, 4] + mole_fracs[1] / bdiff_ij[1, 4] + mole_fracs[2] / bdiff_ij[2, 4] + mole_fracs[3] / bdiff_ij[3, 4] + mole_fracs[4] / bdiff_ij[4, 4] + mole_fracs[5] / bdiff_ij[5, 4] + mole_fracs[6] / bdiff_ij[6, 4],
            mole_fracs[0] / bdiff_ij[0, 5] + mole_fracs[1] / bdiff_ij[1, 5] + mole_fracs[2] / bdiff_ij[2, 5] + mole_fracs[3] / bdiff_ij[3, 5] + mole_fracs[4] / bdiff_ij[4, 5] + mole_fracs[5] / bdiff_ij[5, 5] + mole_fracs[6] / bdiff_ij[6, 5],
            mole_fracs[0] / bdiff_ij[0, 6] + mole_fracs[1] / bdiff_ij[1, 6] + mole_fracs[2] / bdiff_ij[2, 6] + mole_fracs[3] / bdiff_ij[3, 6] + mole_fracs[4] / bdiff_ij[4, 6] + mole_fracs[5] / bdiff_ij[5, 6] + mole_fracs[6] / bdiff_ij[6, 6],
            ])
        denom = self._pyro_make_array([
            x_sum[0] - mole_fracs[0]/bdiff_ij[0, 0],
            x_sum[1] - mole_fracs[1]/bdiff_ij[1, 1],
            x_sum[2] - mole_fracs[2]/bdiff_ij[2, 2],
            x_sum[3] - mole_fracs[3]/bdiff_ij[3, 3],
            x_sum[4] - mole_fracs[4]/bdiff_ij[4, 4],
            x_sum[5] - mole_fracs[5]/bdiff_ij[5, 5],
            x_sum[6] - mole_fracs[6]/bdiff_ij[6, 6],
            ])
        return self._pyro_make_array([
              temperature**(3/2)/pressure*self.usr_np.where(denom[0] > 0,
                  (mmw - mole_fracs[0] * self.wts[0])/(mmw * denom[0]),
                  bdiff_ij[0, 0]
              ),
              temperature**(3/2)/pressure*self.usr_np.where(denom[1] > 0,
                  (mmw - mole_fracs[1] * self.wts[1])/(mmw * denom[1]),
                  bdiff_ij[1, 1]
              ),
              temperature**(3/2)/pressure*self.usr_np.where(denom[2] > 0,
                  (mmw - mole_fracs[2] * self.wts[2])/(mmw * denom[2]),
                  bdiff_ij[2, 2]
              ),
              temperature**(3/2)/pressure*self.usr_np.where(denom[3] > 0,
                  (mmw - mole_fracs[3] * self.wts[3])/(mmw * denom[3]),
                  bdiff_ij[3, 3]
              ),
              temperature**(3/2)/pressure*self.usr_np.where(denom[4] > 0,
                  (mmw - mole_fracs[4] * self.wts[4])/(mmw * denom[4]),
                  bdiff_ij[4, 4]
              ),
              temperature**(3/2)/pressure*self.usr_np.where(denom[5] > 0,
                  (mmw - mole_fracs[5] * self.wts[5])/(mmw * denom[5]),
                  bdiff_ij[5, 5]
              ),
              temperature**(3/2)/pressure*self.usr_np.where(denom[6] > 0,
                  (mmw - mole_fracs[6] * self.wts[6])/(mmw * denom[6]),
                  bdiff_ij[6, 6]
              ),
              ])

    def get_species_specific_heats_r(self, temperature):
        """ Get individual species Cp/R."""
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.03611116 + 0.0146454151*temperature + -6.71077915e-06*temperature**2 + 1.47222923e-09*temperature**3 + -1.25706061e-13*temperature**4, 3.95920148 + -0.00757052247*temperature + 5.70990292e-05*temperature**2 + -6.91588753e-08*temperature**3 + 2.69884373e-11*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.28253784 + 0.00148308754*temperature + -7.57966669e-07*temperature**2 + 2.09470555e-10*temperature**3 + -2.16717794e-14*temperature**4, 3.78245636 + -0.00299673416*temperature + 9.84730201e-06*temperature**2 + -9.68129509e-09*temperature**3 + 3.24372837e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.85746029 + 0.00441437026*temperature + -2.21481404e-06*temperature**2 + 5.23490188e-10*temperature**3 + -4.72084164e-14*temperature**4, 2.35677352 + 0.00898459677*temperature + -7.12356269e-06*temperature**2 + 2.45919022e-09*temperature**3 + -1.43699548e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.71518561 + 0.00206252743*temperature + -9.98825771e-07*temperature**2 + 2.30053008e-10*temperature**3 + -2.03647716e-14*temperature**4, 3.57953347 + -0.00061035368*temperature + 1.01681433e-06*temperature**2 + 9.07005884e-10*temperature**3 + -9.04424499e-13*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.03399249 + 0.00217691804*temperature + -1.64072518e-07*temperature**2 + -9.7041987e-11*temperature**3 + 1.68200992e-14*temperature**4, 4.19864056 + -0.0020364341*temperature + 6.52040211e-06*temperature**2 + -5.48797062e-09*temperature**3 + 1.77197817e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.3372792 + -4.94024731e-05*temperature + 4.99456778e-07*temperature**2 + -1.79566394e-10*temperature**3 + 2.00255376e-14*temperature**4, 2.34433112 + 0.00798052075*temperature + -1.9478151e-05*temperature**2 + 2.01572094e-08*temperature**3 + -7.37611761e-12*temperature**4),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664 + 0.0014879768*temperature + -5.68476e-07*temperature**2 + 1.0097038e-10*temperature**3 + -6.753351e-15*temperature**4, 3.298677 + 0.0014082404*temperature + -3.963222e-06*temperature**2 + 5.641515e-09*temperature**3 + -2.444854e-12*temperature**4),
                ])

    def get_species_enthalpies_rt(self, temperature):
        """ Get individual species h/RT."""
        return self._pyro_make_array([
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.03611116 + 0.00732270755*temperature + -2.2369263833333335e-06*temperature**2 + 3.680573075e-10*temperature**3 + -2.51412122e-14*temperature**4 + 4939.88614 / temperature, 3.95920148 + -0.003785261235*temperature + 1.9033009733333333e-05*temperature**2 + -1.7289718825e-08*temperature**3 + 5.3976874600000004e-12*temperature**4 + 5089.77593 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.28253784 + 0.00074154377*temperature + -2.526555563333333e-07*temperature**2 + 5.236763875e-11*temperature**3 + -4.33435588e-15*temperature**4 + -1088.45772 / temperature, 3.78245636 + -0.00149836708*temperature + 3.282434003333333e-06*temperature**2 + -2.4203237725e-09*temperature**3 + 6.48745674e-13*temperature**4 + -1063.94356 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.85746029 + 0.00220718513*temperature + -7.382713466666667e-07*temperature**2 + 1.30872547e-10*temperature**3 + -9.44168328e-15*temperature**4 + -48759.166 / temperature, 2.35677352 + 0.004492298385*temperature + -2.3745208966666665e-06*temperature**2 + 6.14797555e-10*temperature**3 + -2.8739909599999997e-14*temperature**4 + -48371.9697 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.71518561 + 0.001031263715*temperature + -3.329419236666667e-07*temperature**2 + 5.7513252e-11*temperature**3 + -4.07295432e-15*temperature**4 + -14151.8724 / temperature, 3.57953347 + -0.00030517684*temperature + 3.3893811e-07*temperature**2 + 2.26751471e-10*temperature**3 + -1.808848998e-13*temperature**4 + -14344.086 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.03399249 + 0.00108845902*temperature + -5.469083933333333e-08*temperature**2 + -2.426049675e-11*temperature**3 + 3.36401984e-15*temperature**4 + -30004.2971 / temperature, 4.19864056 + -0.00101821705*temperature + 2.17346737e-06*temperature**2 + -1.371992655e-09*temperature**3 + 3.54395634e-13*temperature**4 + -30293.7267 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.3372792 + -2.470123655e-05*temperature + 1.6648559266666665e-07*temperature**2 + -4.48915985e-11*temperature**3 + 4.00510752e-15*temperature**4 + -950.158922 / temperature, 2.34433112 + 0.003990260375*temperature + -6.4927169999999995e-06*temperature**2 + 5.03930235e-09*temperature**3 + -1.4752235220000002e-12*temperature**4 + -917.935173 / temperature),
            self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664 + 0.0007439884*temperature + -1.8949200000000001e-07*temperature**2 + 2.5242595e-11*temperature**3 + -1.3506701999999999e-15*temperature**4 + -922.7977 / temperature, 3.298677 + 0.0007041202*temperature + -1.3210739999999999e-06*temperature**2 + 1.41037875e-09*temperature**3 + -4.889707999999999e-13*temperature**4 + -1020.8999 / temperature),
                ])

    def get_species_entropies_r(self, temperature):
        """ Get individual species s/R."""
        return self._pyro_make_array([
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.03611116*self.usr_np.log(temperature) + 0.0146454151*temperature + -3.355389575e-06*temperature**2 + 4.907430766666667e-10*temperature**3 + -3.142651525e-14*temperature**4 + 10.3053693, 3.95920148*self.usr_np.log(temperature) + -0.00757052247*temperature + 2.85495146e-05*temperature**2 + -2.3052958433333332e-08*temperature**3 + 6.747109325e-12*temperature**4 + 4.09733096),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.28253784*self.usr_np.log(temperature) + 0.00148308754*temperature + -3.789833345e-07*temperature**2 + 6.982351833333333e-11*temperature**3 + -5.41794485e-15*temperature**4 + 5.45323129, 3.78245636*self.usr_np.log(temperature) + -0.00299673416*temperature + 4.923651005e-06*temperature**2 + -3.2270983633333334e-09*temperature**3 + 8.109320925e-13*temperature**4 + 3.65767573),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.85746029*self.usr_np.log(temperature) + 0.00441437026*temperature + -1.10740702e-06*temperature**2 + 1.7449672933333335e-10*temperature**3 + -1.18021041e-14*temperature**4 + 2.27163806, 2.35677352*self.usr_np.log(temperature) + 0.00898459677*temperature + -3.561781345e-06*temperature**2 + 8.197300733333333e-10*temperature**3 + -3.5924887e-14*temperature**4 + 9.90105222),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.71518561*self.usr_np.log(temperature) + 0.00206252743*temperature + -4.994128855e-07*temperature**2 + 7.6684336e-11*temperature**3 + -5.0911929e-15*temperature**4 + 7.81868772, 3.57953347*self.usr_np.log(temperature) + -0.00061035368*temperature + 5.08407165e-07*temperature**2 + 3.023352946666667e-10*temperature**3 + -2.2610612475e-13*temperature**4 + 3.50840928),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.03399249*self.usr_np.log(temperature) + 0.00217691804*temperature + -8.2036259e-08*temperature**2 + -3.2347329e-11*temperature**3 + 4.2050248e-15*temperature**4 + 4.9667701, 4.19864056*self.usr_np.log(temperature) + -0.0020364341*temperature + 3.260201055e-06*temperature**2 + -1.82932354e-09*temperature**3 + 4.429945425e-13*temperature**4 + -0.849032208),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 3.3372792*self.usr_np.log(temperature) + -4.94024731e-05*temperature + 2.49728389e-07*temperature**2 + -5.985546466666667e-11*temperature**3 + 5.0063844e-15*temperature**4 + -3.20502331, 2.34433112*self.usr_np.log(temperature) + 0.00798052075*temperature + -9.7390755e-06*temperature**2 + 6.7190698e-09*temperature**3 + -1.8440294025e-12*temperature**4 + 0.683010238),
                self.usr_np.where(self.usr_np.greater(temperature, 1000.0), 2.92664*self.usr_np.log(temperature) + 0.0014879768*temperature + -2.84238e-07*temperature**2 + 3.3656793333333334e-11*temperature**3 + -1.68833775e-15*temperature**4 + 5.980528, 3.298677*self.usr_np.log(temperature) + 0.0014082404*temperature + -1.981611e-06*temperature**2 + 1.8805050000000002e-09*temperature**3 + -6.112135e-13*temperature**4 + 3.950372),
                ])

    def get_species_gibbs_rt(self, temperature):
        """ Get individual species G/RT."""
        h0_rt = self.get_species_enthalpies_rt(temperature)
        s0_r = self.get_species_entropies_r(temperature)
        return h0_rt - s0_r

    def get_equilibrium_constants(self, temperature):
        rt = self.gas_constant * temperature
        c0 = self.usr_np.log(self.one_atm / rt)

        g0_rt = self.get_species_gibbs_rt(temperature)
        return self._pyro_make_array([
                    -0.17364695002734*temperature,
                    g0_rt[2] + -1*(g0_rt[3] + 0.5*g0_rt[1]) + -1*-0.5*c0,
                    -0.17364695002734*temperature,
                ])

    def get_temperature(self, enthalpy_or_energy, t_guess, y, do_energy=False):
        if do_energy is False:
            pv_fun = self.get_mixture_specific_heat_cp_mass
            he_fun = self.get_mixture_enthalpy_mass
        else:
            pv_fun = self.get_mixture_specific_heat_cv_mass
            he_fun = self.get_mixture_internal_energy_mass

        num_iter = 500
        tol = 1.0e-6
        ones = self._pyro_zeros_like(enthalpy_or_energy) + 1.0
        t_i = t_guess * ones

        for _ in range(num_iter):
            f = enthalpy_or_energy - he_fun(t_i, y)
            j = -pv_fun(t_i, y)
            dt = -f / j
            t_i += dt
            if self._pyro_norm(dt, np.inf) < tol:
                return t_i

        raise RuntimeError("Temperature iteration failed to converge")

    def get_fwd_rate_coefficients(self, temperature, concentrations):
        ones = self._pyro_zeros_like(temperature) + 1.0
        k_fwd = [
            self.usr_np.exp(21.989687638093134 + -1*(18115.903205955565 / temperature)) * ones,
            self.usr_np.exp(12.759528191271578 + 0.7*self.usr_np.log(temperature) + -1*(5535.414868486423 / temperature)) * ones,
            self.usr_np.exp(18.639652073262145 + -1*(6038.634401985189 / temperature)) * ones,
                ]

        return self._pyro_make_array(k_fwd)

    def get_net_rates_of_progress(self, temperature, concentrations):
        k_fwd = self.get_fwd_rate_coefficients(temperature, concentrations)
        log_k_eq = self.get_equilibrium_constants(temperature)
        return self._pyro_make_array([
                    k_fwd[0]*concentrations[0]**0.5*concentrations[1]**0.65,
                    k_fwd[1]*(concentrations[3]*concentrations[1]**0.5 + -1*self.usr_np.exp(log_k_eq[1])*concentrations[2]),
                    k_fwd[2]*concentrations[5]**0.75*concentrations[1]**0.5,
               ])

    def get_net_production_rates(self, rho, temperature, mass_fractions, concentrations=None):
        if concentrations is None:
            concentrations = self.get_concentrations(rho, mass_fractions)
        r_net = self.get_net_rates_of_progress(temperature, concentrations)
        ones = self._pyro_zeros_like(r_net[0]) + 1.0
        return self._pyro_make_array([
                -1*r_net[0] * ones,
                -1*(r_net[0] + 0.5*r_net[1] + 0.5*r_net[2]) * ones,
                r_net[1] * ones,
                2.0*r_net[0] + -1*r_net[1] * ones,
                r_net[2] * ones,
                2.0*r_net[0] + -1*r_net[2] * ones,
                0.0 * ones,
               ])
