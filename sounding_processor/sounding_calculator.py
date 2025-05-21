# sounding_processor/sounding_processor/sounding_calculator.py
"""
Calculates derived meteorological parameters for a given Sounding object.
"""
import logging
import numpy as np
import pandas as pd
import traceback 
from typing import Dict, Any, Tuple, Optional 

from .sounding_data import Sounding
from .metpy_manager import metpy_manager_instance as mpi 
from . import config as app_config 

logger = logging.getLogger(__name__)

class SoundingCalculator:
    """
    Calculates derived parameters for a Sounding object.
    Relies on MetPyManager for MetPy functions and units.
    """
    def __init__(self):
        if not mpi.metpy_available:
            logger.critical("MetPy is not available. SoundingCalculator will not be able to perform most calculations.")
        self.mpcalc = mpi.get_calculator() 
        self.units = mpi.get_units()       
        if self.units:
            self.energy_per_mass_units = self.units.joule / self.units.kilogram
            self.specific_energy_rate_units = self.units.m / (self.units.s**2) 
        else:
            self.energy_per_mass_units = None 
            self.specific_energy_rate_units = None


    def _prepare_profile_data(self, sounding: Sounding) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
        if not mpi.metpy_available or self.mpcalc is None or self.units is None:
            logger.error("MetPy components not available for data preparation.")
            return None, sounding.level_data 

        level_df_orig = sounding.level_data
        if not isinstance(level_df_orig, pd.DataFrame) or level_df_orig.empty:
            logger.warning(f"Sounding {sounding.metadata.get('STID','N/A')}/{sounding.metadata.get('TIME', 'N/A')} has no level_data DataFrame.")
            return None, None 

        level_df = level_df_orig.copy()
        essential_cols = ['PRES', 'HGHT', 'TMPC', 'DWPC', 'SKNT', 'DRCT']
        missing_cols = [col for col in essential_cols if col not in level_df.columns]
        if missing_cols:
            logger.error(f"Essential columns {missing_cols} missing in level_data for sounding {sounding.metadata.get('STID','N/A')}/{sounding.metadata.get('TIME', 'N/A')}.")
            return None, level_df 

        for col in essential_cols:
            level_df[col] = pd.to_numeric(level_df[col], errors='coerce')
        
        level_df.dropna(subset=essential_cols, inplace=True)
        level_df = level_df.sort_values(by='PRES', ascending=False).reset_index(drop=True)

        if level_df.empty or len(level_df) < 2: 
            logger.warning(f"Level_df empty or <2 levels after cleaning for sounding {sounding.metadata.get('STID','N/A')}/{sounding.metadata.get('TIME', 'N/A')}.")
            return None, level_df 

        try:
            p = level_df['PRES'].values * self.units.hPa
            T = level_df['TMPC'].values * self.units.degC
            Td = level_df['DWPC'].values * self.units.degC
            hght_msl = level_df['HGHT'].values * self.units.meter
            
            sfc_h_msl = hght_msl[0] if hght_msl.size > 0 else (0 * self.units.meter)
            hght_agl_calc = hght_msl - sfc_h_msl
            hght_agl_mag = np.maximum(hght_agl_calc.magnitude, 0) 
            hght_agl = hght_agl_mag * self.units.meter

            sknt = level_df['SKNT'].values * self.units.knots
            drct = level_df['DRCT'].values * self.units.degrees
            
            u = np.full(len(sknt), np.nan) * self.units.knots 
            v = np.full(len(sknt), np.nan) * self.units.knots 

            if hasattr(self.mpcalc, 'wind_components'):
                valid_wind_mask = pd.notna(sknt.magnitude) & pd.notna(drct.magnitude)
                if np.any(valid_wind_mask):
                    sknt_valid = sknt[valid_wind_mask]
                    drct_valid = drct[valid_wind_mask]
                    u_valid, v_valid = self.mpcalc.wind_components(sknt_valid, drct_valid)
                    u[valid_wind_mask] = u_valid
                    v[valid_wind_mask] = v_valid
            else:
                logger.warning("mpcalc.wind_components not found. U/V components will be NaN.")

            profile_data = {
                'p': p, 'T': T, 'Td': Td,
                'hght_msl': hght_msl, 'hght_agl': hght_agl,
                'sfc_h_msl': sfc_h_msl,
                'u': u, 'v': v, 'sknt': sknt, 'drct': drct
            }
            return profile_data, level_df 

        except Exception as e_units: 
            logger.error(f"Error assigning units or calculating components for {sounding.metadata.get('STID','N/A')}: {e_units}", exc_info=True)
            return None, level_df 

    def calculate_all_derived_params(self, sounding: Sounding) -> Sounding:
        current_time_id = f"STID {sounding.metadata.get('STID', 'N/A')} TIME {sounding.metadata.get('TIME', 'N/A')}"
        logger.info(f"Starting derived parameter calculation for {current_time_id}")

        if not mpi.metpy_available or self.mpcalc is None or self.units is None or self.energy_per_mass_units is None:
            logger.error(f"MetPy components (or derived units) not available. Cannot calculate derived params for {current_time_id}.")
            sounding.derived_params_calculated = False 
            return sounding

        profile_data_dict, cleaned_level_df = self._prepare_profile_data(sounding)
        if cleaned_level_df is not None: 
            sounding.level_data = cleaned_level_df

        if profile_data_dict is None:
            logger.warning(f"Profile data preparation failed for {current_time_id}. Skipping calculations.")
            sounding.derived_params_calculated = False
            return sounding
        
        sounding.metadata['_internal_profile_data'] = profile_data_dict

        try:
            self._calculate_parcel_thermodynamics(sounding)
            self._calculate_detailed_kinematics(sounding)
            self._calculate_environmental_thermodynamics(sounding)
            self._calculate_key_levels_snapshot(sounding) 
            self._calculate_composite_indices(sounding)
            self._calculate_inversion_characteristics(sounding)
            
            sounding.derived_params_calculated = True
            logger.info(f"Successfully calculated derived parameters for {current_time_id}")

        except Exception as e_calc_all: 
            logger.error(f"Unhandled error during calculation suite for {current_time_id}: {e_calc_all}", exc_info=True)
            sounding.derived_params_calculated = False
            for key in ['parcel_thermodynamics', 'detailed_kinematics', 'environmental_thermodynamics',
                        'composite_indices', 'inversion_characteristics', 'key_levels_data']:
                if not hasattr(sounding, key) or getattr(sounding, key) is None: 
                    setattr(sounding, key, {})
        finally:
            if '_internal_profile_data' in sounding.metadata:
                del sounding.metadata['_internal_profile_data']

        return sounding

    def _get_prof_data(self, sounding: Sounding) -> Dict[str, Any]:
        data = sounding.metadata.get('_internal_profile_data')
        if data is None:
            logger.critical("Internal profile data not found on sounding object.")
            raise ValueError("Missing internal profile data for calculations.")
        return data

    def _calculate_parcel_thermodynamics(self, sounding: Sounding):
        prof_data = self._get_prof_data(sounding)
        pt_dict = sounding.parcel_thermodynamics 
        p, T, Td, hght_agl = prof_data['p'], prof_data['T'], prof_data['Td'], prof_data['hght_agl']
        logger.debug(f"Calculating parcel thermo for {sounding.metadata.get('STID','N/A')}")

        sfc_p, sfc_T, sfc_Td = p[0], T[0], Td[0] 
        sfc_lfc_p, sfc_lfc_T, sfc_el_p, sfc_el_T = [np.nan * self.units.hPa]*2 + [np.nan * self.units.degC]*2
        ml_lfc_p, ml_lfc_T, ml_el_p, ml_el_T = [np.nan * self.units.hPa]*2 + [np.nan * self.units.degC]*2
        mu_lfc_p, mu_lfc_T, mu_el_p, mu_el_T = [np.nan * self.units.hPa]*2 + [np.nan * self.units.degC]*2
        
        sfc_prof = np.nan * self.units.degC
        if hasattr(self.mpcalc, 'parcel_profile'):
            try: sfc_prof = self.mpcalc.parcel_profile(p, sfc_T, sfc_Td).to(self.units.degC)
            except Exception as e: logger.warning(f"SFC parcel_profile: {e}", exc_info=True)
        
        sfc_cape_q, sfc_cin_q = np.nan * self.energy_per_mass_units, np.nan * self.energy_per_mass_units
        if hasattr(self.mpcalc, 'cape_cin') and hasattr(sfc_prof, 'units') and sfc_prof.size == p.size:
            try: sfc_cape_q, sfc_cin_q = self.mpcalc.cape_cin(p, T, Td, sfc_prof)
            except Exception as e: logger.warning(f"SFC cape_cin: {e}", exc_info=True)
        pt_dict['sfc_cape_jkg'] = mpi._round_quantity_magnitude(sfc_cape_q, 1)
        pt_dict['sfc_cin_jkg'] = mpi._round_quantity_magnitude(sfc_cin_q, 1)

        if hasattr(sfc_prof, 'units') and sfc_prof.size == p.size:
            if hasattr(self.mpcalc, 'lfc'):
                try: sfc_lfc_p, sfc_lfc_T = self.mpcalc.lfc(p, T, Td, parcel_temperature_profile=sfc_prof)
                except Exception as e: logger.warning(f"SFC LFC: {e}", exc_info=True)
            if hasattr(self.mpcalc, 'el'):
                try: sfc_el_p, sfc_el_T = self.mpcalc.el(p, T, Td, parcel_temperature_profile=sfc_prof)
                except Exception as e: logger.warning(f"SFC EL: {e}", exc_info=True)
        
        ml_depth = app_config.ML_DEPTH_HPA * self.units.hPa
        ml_p_parcel, ml_T_parcel, ml_Td_parcel = (np.nan * self.units.hPa, np.nan * self.units.degC, np.nan * self.units.degC)
        ml_prof = np.nan * self.units.degC
        if hasattr(self.mpcalc, 'mixed_parcel') and hasattr(self.mpcalc, 'parcel_profile'):
            try:
                ml_p_parcel, ml_T_parcel, ml_Td_parcel = self.mpcalc.mixed_parcel(p, T, Td, depth=ml_depth)
                if all(pd.notna(getattr(val, 'magnitude', val)) for val in [ml_T_parcel, ml_Td_parcel]):
                    ml_prof = self.mpcalc.parcel_profile(p, ml_T_parcel, ml_Td_parcel).to(self.units.degC)
            except Exception as e: logger.warning(f"ML mixed_parcel or parcel_profile: {e}", exc_info=True)
        
        _mlcape, _mlcin = np.nan * self.energy_per_mass_units, np.nan * self.energy_per_mass_units
        if hasattr(self.mpcalc, 'mixed_layer_cape_cin'):
            try: _mlcape, _mlcin = self.mpcalc.mixed_layer_cape_cin(p,T,Td,depth=ml_depth)
            except Exception as e: logger.warning(f"mixed_layer_cape_cin: {e}", exc_info=True)
        elif hasattr(self.mpcalc, 'cape_cin') and hasattr(ml_prof, 'units') and ml_prof.size == p.size :
            try: _mlcape, _mlcin = self.mpcalc.cape_cin(p, T, Td, ml_prof)
            except Exception as e: logger.warning(f"ML cape_cin (fallback): {e}", exc_info=True)
        pt_dict['mlcape_100hpa_jkg'] = mpi._round_quantity_magnitude(_mlcape, 1)
        pt_dict['mlcin_100hpa_jkg'] = mpi._round_quantity_magnitude(_mlcin, 1)

        if hasattr(ml_prof, 'units') and ml_prof.size == p.size :
            if hasattr(self.mpcalc, 'lfc'):
                try: ml_lfc_p, ml_lfc_T = self.mpcalc.lfc(p, T, Td, parcel_temperature_profile=ml_prof)
                except Exception as e: logger.warning(f"ML LFC: {e}", exc_info=True)
            if hasattr(self.mpcalc, 'el'):
                try: ml_el_p, ml_el_T = self.mpcalc.el(p, T, Td, parcel_temperature_profile=ml_prof)
                except Exception as e: logger.warning(f"ML EL: {e}", exc_info=True)
        
        mup, muT, muTd, _ = (np.nan * self.units.hPa, np.nan * self.units.degC, np.nan * self.units.degC, np.nan)
        mu_prof = np.nan * self.units.degC
        if hasattr(self.mpcalc, 'most_unstable_parcel') and hasattr(self.mpcalc, 'parcel_profile'):
            try:
                if hght_agl.size == p.size:
                    mup, muT, muTd, _ = self.mpcalc.most_unstable_parcel(p, T, Td, height=hght_agl)
                    if all(pd.notna(getattr(val, 'magnitude', val)) for val in [muT, muTd]):
                         mu_prof = self.mpcalc.parcel_profile(p, muT, muTd).to(self.units.degC)
                else: logger.warning(f"MUP: hght_agl size mismatch p size. h_agl:{hght_agl.size}, p:{p.size}")
            except Exception as e: logger.warning(f"MU most_unstable_parcel or parcel_profile: {e}", exc_info=True)
        
        mucape, mucin = np.nan * self.energy_per_mass_units, np.nan * self.energy_per_mass_units
        if hasattr(self.mpcalc, 'most_unstable_cape_cin'):
            try:
                if hght_agl.size == p.size: mucape, mucin = self.mpcalc.most_unstable_cape_cin(p,T,Td,height=hght_agl)
                else: mucape, mucin = self.mpcalc.most_unstable_cape_cin(p,T,Td)
            except Exception as e: logger.warning(f"most_unstable_cape_cin: {e}", exc_info=True)
        elif hasattr(self.mpcalc, 'cape_cin') and hasattr(mu_prof, 'units') and mu_prof.size == p.size :
            try: mucape, mucin = self.mpcalc.cape_cin(p, T, Td, mu_prof)
            except Exception as e: logger.warning(f"MU cape_cin (fallback): {e}", exc_info=True)
        pt_dict['mucape_jkg'] = mpi._round_quantity_magnitude(mucape, 1)
        pt_dict['mucin_jkg'] = mpi._round_quantity_magnitude(mucin, 1)

        if hasattr(mu_prof, 'units') and mu_prof.size == p.size :
            if hasattr(self.mpcalc, 'lfc'):
                try: mu_lfc_p, mu_lfc_T = self.mpcalc.lfc(p, T, Td, parcel_temperature_profile=mu_prof)
                except Exception as e: logger.warning(f"MU LFC: {e}", exc_info=True)
            if hasattr(self.mpcalc, 'el'):
                try: mu_el_p, mu_el_T = self.mpcalc.el(p, T, Td, parcel_temperature_profile=mu_prof)
                except Exception as e: logger.warning(f"MU EL: {e}", exc_info=True)

        interp_method_log = 'log' if app_config.METPY_LOG_INTERPOLATE_1D_AVAILABLE else 'numpy_log'
        if hasattr(self.mpcalc, 'lcl'):
            try:
                lcl_p, lcl_T = self.mpcalc.lcl(sfc_p, sfc_T, sfc_Td)
                pt_dict['sfc_lcl_p_hpa'] = mpi._round_quantity_magnitude(lcl_p, 1, 'hPa')
                pt_dict['sfc_lcl_t_c'] = mpi._round_quantity_magnitude(lcl_T, 1, 'degC')
                lcl_h = mpi.interpolate_profile_value(lcl_p, p, hght_agl, interp_method_log)
                pt_dict['sfc_lcl_h_m_agl'] = mpi._round_quantity_magnitude(lcl_h, 0, 'meter')
            except Exception as e: logger.warning(f"SFC LCL: {e}", exc_info=True)
        
        pt_dict['mu_lfc_p_hpa'] = mpi._round_quantity_magnitude(mu_lfc_p, 1, 'hPa')
        pt_dict['mu_lfc_t_c'] = mpi._round_quantity_magnitude(mu_lfc_T, 1, 'degC')
        mu_lfc_h = mpi.interpolate_profile_value(mu_lfc_p, p, hght_agl, interp_method_log)
        pt_dict['mu_lfc_h_m_agl'] = mpi._round_quantity_magnitude(mu_lfc_h, 0, 'meter')
        pt_dict['mu_el_p_hpa'] = mpi._round_quantity_magnitude(mu_el_p, 1, 'hPa')
        pt_dict['mu_el_t_c'] = mpi._round_quantity_magnitude(mu_el_T, 1, 'degC')
        mu_el_h = mpi.interpolate_profile_value(mu_el_p, p, hght_agl, interp_method_log)
        pt_dict['mu_el_h_m_agl'] = mpi._round_quantity_magnitude(mu_el_h, 0, 'meter')

        if hasattr(mucape, 'm') and pd.notna(mucape.m) and mucape.m > 0 and \
           hasattr(mu_el_h, 'm') and pd.notna(mu_el_h.m) and \
           hasattr(mu_lfc_h, 'm') and pd.notna(mu_lfc_h.m) and \
           mu_el_h > mu_lfc_h: 
            try:
                cape_depth = (mu_el_h - mu_lfc_h).to('meter')
                if hasattr(cape_depth,'m') and pd.notna(cape_depth.m) and cape_depth.m > 1e-6 and self.specific_energy_rate_units:
                    
                    # --- MODIFICATION START ---
                    # Ensure mucape and cape_depth are scalar quantities or simple floats before division
                    # to maintain Quantity type integrity or handle potential raw numpy arrays.
                    current_mucape = mucape
                    current_cape_depth = cape_depth

                    if hasattr(current_mucape, 'item') and hasattr(current_mucape, 'ndim') and current_mucape.ndim == 0:
                        current_mucape = current_mucape.item() # Should return scalar Quantity if input was 0-d Quantity
                    elif isinstance(current_mucape, np.ndarray) and current_mucape.ndim == 0: # Raw numpy 0-d array
                        current_mucape = current_mucape.item() * self.energy_per_mass_units # Re-attach units
                    
                    if hasattr(current_cape_depth, 'item') and hasattr(current_cape_depth, 'ndim') and current_cape_depth.ndim == 0:
                        current_cape_depth = current_cape_depth.item()
                    elif isinstance(current_cape_depth, np.ndarray) and current_cape_depth.ndim == 0:
                        current_cape_depth = current_cape_depth.item() * self.units.meter
                    
                    # Check if they are still quantities after .item() or re-attachment
                    if not (hasattr(current_mucape, 'units') and hasattr(current_cape_depth, 'units')):
                        logger.warning(f"NCAPE: Operands lost units. MUCAPE type: {type(current_mucape)}, Depth type: {type(current_cape_depth)}")
                        ncape_value = np.nan * self.specific_energy_rate_units # Cannot reliably calculate
                    else:
                        ncape_value = current_mucape / current_cape_depth
                    # --- MODIFICATION END ---
                                        
                    logger.debug(f"NCAPE calc: mucape type={type(current_mucape)}, mucape units={getattr(current_mucape,'units','N/A')}, cape_depth type={type(current_cape_depth)}, cape_depth units={getattr(current_cape_depth,'units','N/A')}, ncape_value before round type={type(ncape_value)}, units={getattr(ncape_value, 'units', 'N/A')}, value={ncape_value}")
                    pt_dict['mu_ncape_s_neg2'] = mpi._round_quantity_magnitude(ncape_value, 4, str(self.specific_energy_rate_units))
            except Exception as e: logger.warning(f"NCAPE: {e}", exc_info=True)
        
        if hasattr(self.mpcalc, 'downdraft_cape'):
            try:
                dcape_q, _, _ = self.mpcalc.downdraft_cape(p, T, Td)
                pt_dict['dcape_jkg'] = mpi._round_quantity_magnitude(dcape_q, 1)
            except Exception as e: logger.warning(f"DCAPE: {e}", exc_info=True)
        
        sounding.metadata['_intermediate_mup_p'] = mup
        sounding.metadata['_intermediate_mup_T'] = muT
        sounding.metadata['_intermediate_mup_Td'] = muTd
        
    def _calculate_detailed_kinematics(self, sounding: Sounding):
        prof_data = self._get_prof_data(sounding)
        dk_dict = sounding.detailed_kinematics
        p, u, v, hght_agl = prof_data['p'], prof_data['u'], prof_data['v'], prof_data['hght_agl']
        logger.debug(f"Calculating kinematics for {sounding.metadata.get('STID','N/A')}")

        valid_mask = pd.notna(p.magnitude)&pd.notna(u.magnitude)&pd.notna(v.magnitude)&pd.notna(hght_agl.magnitude)
        if not np.any(valid_mask) or np.sum(valid_mask) < 2: 
            logger.warning("Not enough valid data points for kinematic calculations.")
            return

        p_k, u_k, v_k, h_k = p[valid_mask], u[valid_mask], v[valid_mask], hght_agl[valid_mask]
        sort_idx = np.argsort(h_k.magnitude) 
        p_s, u_s, v_s, h_s = p_k[sort_idx], u_k[sort_idx], v_k[sort_idx], h_k[sort_idx]
        max_h = h_s[-1] if h_s.size > 0 else 0 * self.units.meter

        rmu,rmv,lmu,lmv,mwu,mwv = [np.nan*self.units.mps]*6
        if hasattr(self.mpcalc, 'bunkers_storm_motion'):
            try:
                # ADD DEBUG LOGS HERE
                logger.debug(f"Bunkers Input p_s ({p_s.units}): {p_s.magnitude[:5]}...{p_s.magnitude[-5:] if p_s.size > 5 else ''}")
                logger.debug(f"Bunkers Input u_s ({u_s.units}): {u_s.magnitude[:5]}...{u_s.magnitude[-5:] if u_s.size > 5 else ''}")
                logger.debug(f"Bunkers Input v_s ({v_s.units}): {v_s.magnitude[:5]}...{v_s.magnitude[-5:] if v_s.size > 5 else ''}")
                logger.debug(f"Bunkers Input h_s ({h_s.units}): {h_s.magnitude[:5]}...{h_s.magnitude[-5:] if h_s.size > 5 else ''}")
                
                rm, lm, mw = self.mpcalc.bunkers_storm_motion(p_s, u_s, v_s, h_s) 
                if rm[0].size>0 and pd.notna(rm[0].m): rmu,rmv=rm
                if lm[0].size>0 and pd.notna(lm[0].m): lmu,lmv=lm
                if mw[0].size>0 and pd.notna(mw[0].m): mwu,mwv=mw
            except Exception as e: logger.warning(f"Bunkers storm motion: {e}", exc_info=True)
        
        mw06u,mw06v = mwu,mwv; mw_src = "bunkers_internal_mean"
        if not (hasattr(mw06u,'m') and pd.notna(mw06u.m)):
            d6k = min(6000*self.units.m, max_h)
            if hasattr(d6k, 'm') and pd.notna(d6k.m) and d6k.m > 500: 
                if hasattr(self.mpcalc, 'mean_wind'): 
                    try: mw06u,mw06v = self.mpcalc.mean_wind(h_s, u_s, v_s, depth=d6k); mw_src="metpy_mw_0_6km"
                    except Exception as e: logger.warning(f"MetPy mean_wind (0-6km): {e}", exc_info=True)
                elif hasattr(mpi, '_manual_mean_wind'): 
                    try: mw06u,mw06v = mpi._manual_mean_wind(h_s,u_s,v_s,depth_q=d6k); mw_src="manual_mw_0_6km"
                    except Exception as e: logger.warning(f"Manual mean_wind (0-6km): {e}", exc_info=True)
                else: mw_src="unavailable_mean_wind_func"
            else: mw_src="depth_0_6km_too_shallow_or_nan"
        
        for prefix, val_u, val_v in [
            ('bunkers_right_mover', rmu, rmv), ('bunkers_left_mover', lmu, lmv),
            ('bunkers_mean_wind', mwu, mwv), ('mean_wind_0_6km', mw06u, mw06v)
        ]:
            dk_dict[f'{prefix}_u_kts'] = mpi._round_quantity_magnitude(val_u,1,'knots')
            dk_dict[f'{prefix}_v_kts'] = mpi._round_quantity_magnitude(val_v,1,'knots')
        dk_dict['mean_wind_source_0_6km'] = mw_src

        if hasattr(self.mpcalc, 'bulk_shear') and hasattr(self.mpcalc, 'wind_speed'):
            for tag, depth_m_val in [('1km',1000), ('6km',6000)]:
                key_out = f'bulk_shear_0_{tag}_kts'
                actual_depth_q = min(depth_m_val*self.units.m, max_h)
                min_req_depth_m = 100 if depth_m_val==1000 else 300 
                if hasattr(actual_depth_q,'m') and pd.notna(actual_depth_q.m) and actual_depth_q.m >= min_req_depth_m:
                    try:
                        bsu,bsv = self.mpcalc.bulk_shear(p_s, u_s, v_s, height=h_s, depth=actual_depth_q)
                        dk_dict[key_out] = mpi._round_quantity_magnitude(self.mpcalc.wind_speed(bsu,bsv),1,'knots')
                    except Exception as e: 
                        logger.warning(f"Bulk shear 0-{tag}: {e}", exc_info=True)
                        dk_dict[key_out]=np.nan
                else: dk_dict[key_out]=np.nan
        
        smu,smv = rmu,rmv 
        if not (hasattr(smu,'m') and pd.notna(smu.m)): smu,smv = mw06u,mw06v 
        if not (hasattr(smu,'m') and pd.notna(smu.m)) and u_s.size>0: smu,smv = u_s[0],v_s[0] 

        if hasattr(self.mpcalc, 'storm_relative_helicity') and hasattr(smu,'m') and pd.notna(smu.m):
            for tag, d_m_val in app_config.SRH_CALC_DEPTHS_M.items(): 
                key_out = f'srh_{tag}_m2s2_eff_sm'
                actual_srh_depth_q = min(d_m_val*self.units.m, max_h)
                min_req_srh_depth_m = 100 if d_m_val==1000 else 300
                if hasattr(actual_srh_depth_q,'m') and pd.notna(actual_srh_depth_q.m) and actual_srh_depth_q.m >= min_req_srh_depth_m:
                    try:
                        srh_q,_,_ = self.mpcalc.storm_relative_helicity(h_s,u_s,v_s,depth=actual_srh_depth_q,bottom=0*self.units.m,storm_u=smu,storm_v=smv)
                        dk_dict[key_out] = mpi._round_quantity_magnitude(srh_q,1) 
                    except Exception as e: 
                        logger.warning(f"SRH {tag}: {e}", exc_info=True)
                        dk_dict[key_out]=np.nan
                else: dk_dict[key_out]=np.nan
        
        for eil_key in ['effective_inflow_layer_bottom_m_agl', 'effective_inflow_layer_top_m_agl',
                        'effective_bulk_wind_difference_kts', 'effective_srh_m2s2',
                        'effective_layer_mean_wind_u_kts', 'effective_layer_mean_wind_v_kts']:
            dk_dict[eil_key] = np.nan 
        if hasattr(self.mpcalc, 'effective_inflow_layer'): logger.info("MetPy function 'effective_inflow_layer' is available. Consider implementing EIL calculations.")

        mw01u,mw01v = np.nan*self.units.mps, np.nan*self.units.mps; mw01_src="unavailable"
        d1k = min(1000*self.units.m,max_h)
        if hasattr(d1k,'m') and pd.notna(d1k.m) and d1k.m > 100:
            if hasattr(self.mpcalc, 'mean_wind'):
                try: mw01u,mw01v=self.mpcalc.mean_wind(h_s,u_s,v_s,depth=d1k); mw01_src="metpy_mw_0_1km"
                except Exception as e: logger.warning(f"MetPy mean_wind (0-1km): {e}", exc_info=True)
            elif hasattr(mpi, '_manual_mean_wind'):
                try: mw01u,mw01v=mpi._manual_mean_wind(h_s,u_s,v_s,depth_q=d1k); mw01_src="manual_mw_0_1km"
                except Exception as e: logger.warning(f"Manual mean_wind (0-1km): {e}", exc_info=True)
        dk_dict['mean_wind_0_1km_u_kts']=mpi._round_quantity_magnitude(mw01u,1,'knots')
        dk_dict['mean_wind_0_1km_v_kts']=mpi._round_quantity_magnitude(mw01v,1,'knots')
        dk_dict['mean_wind_source_0_1km']=mw01_src

    def _calculate_environmental_thermodynamics(self, sounding: Sounding):
        prof_data = self._get_prof_data(sounding)
        et_dict = sounding.environmental_thermodynamics
        p,T,Td,h_agl,h_msl = prof_data['p'],prof_data['T'],prof_data['Td'],prof_data['hght_agl'],prof_data['hght_msl']
        sfc_p, sfc_T, sfc_Td = p[0], T[0], Td[0] 
        logger.debug(f"Calculating env thermo for {sounding.metadata.get('STID','N/A')}")

        if hasattr(self.mpcalc, 'precipitable_water') and p.size>=2 and np.any(pd.notna(Td.magnitude)):
            try: et_dict['precipitable_water_mm'] = mpi._round_quantity_magnitude(self.mpcalc.precipitable_water(p,Td),2,'mm')
            except Exception as e: logger.warning(f"PWAT: {e}", exc_info=True)
        
        sfc_prof_li = np.nan * self.units.degC
        if hasattr(self.mpcalc,'parcel_profile'):
            try: sfc_prof_li = self.mpcalc.parcel_profile(p,sfc_T,sfc_Td).to(self.units.degC)
            except Exception: pass
        if hasattr(self.mpcalc,'lifted_index') and hasattr(sfc_prof_li,'units') and sfc_prof_li.size==p.size:
            try: 
                li_result = self.mpcalc.lifted_index(p,T,sfc_prof_li)
                li_val_to_round = li_result[0] if isinstance(li_result, tuple) else li_result
                et_dict['lifted_index_c']=mpi._round_quantity_magnitude(li_val_to_round,2,'delta_degC')
            except Exception as e: logger.warning(f"LI: {e}", exc_info=True)

        interp_lin = 'linear' if app_config.METPY_INTERPOLATE_1D_AVAILABLE else 'numpy'
        interp_log = 'log' if app_config.METPY_LOG_INTERPOLATE_1D_AVAILABLE else 'numpy_log'
        
        sort_h_idx = np.argsort(h_agl.magnitude); h_s_h, T_s_h = h_agl[sort_h_idx], T[sort_h_idx]
        sort_p_idx = np.argsort(p.magnitude); p_s_p, T_s_p, h_msl_s_p = p[sort_p_idx], T[sort_p_idx], h_msl[sort_p_idx]

        def get_lr_h(h_q,T_q,b,t,ip):
            if not(h_q.size>1 and T_q.size==h_q.size and t>b): return np.nan*self.units('delta_degC/km')
            bq,tq=b*self.units.m,t*self.units.m
            Tb=mpi.interpolate_profile_value(bq,h_q,T_q,ip); Tt=mpi.interpolate_profile_value(tq,h_q,T_q,ip)
            if any(pd.isna(getattr(v,'m',v)) for v in [Tb,Tt]): return np.nan*self.units('delta_degC/km')
            dH=(tq-bq).to('km')
            if hasattr(dH,'m') and pd.notna(dH.m) and abs(dH.m)>1e-6: return (Tt-Tb)/dH
            return np.nan*self.units('delta_degC/km')
        et_dict['lapse_rate_sfc_1km_c_km'] = mpi._round_quantity_magnitude(get_lr_h(h_s_h,T_s_h,0,1000,interp_lin),2)
        et_dict['lapse_rate_sfc_3km_c_km'] = mpi._round_quantity_magnitude(get_lr_h(h_s_h,T_s_h,0,3000,interp_lin),2)

        def get_lr_p(p_q,T_q,h_msl_q,bp,tp,ip):
            if not(p_q.size>1 and T_q.size==p_q.size and h_msl_q.size==p_q.size and bp>tp): return np.nan*self.units('delta_degC/km')
            bpq,tpq = bp*self.units.hPa, tp*self.units.hPa
            Tb=mpi.interpolate_profile_value(bpq,p_q,T_q,ip); Tt=mpi.interpolate_profile_value(tpq,p_q,T_q,ip)
            Hb=mpi.interpolate_profile_value(bpq,p_q,h_msl_q,ip); Ht=mpi.interpolate_profile_value(tpq,p_q,h_msl_q,ip)
            if any(pd.isna(getattr(v,'m',v)) for v in [Tb,Tt,Hb,Ht]): return np.nan*self.units('delta_degC/km')
            dH=(Ht-Hb).to('km')
            if hasattr(dH,'m') and pd.notna(dH.m) and abs(dH.m)>1e-6: return (Tt-Tb)/dH
            return np.nan*self.units('delta_degC/km')
        et_dict['lapse_rate_850_500mb_c_km']=mpi._round_quantity_magnitude(get_lr_p(p_s_p,T_s_p,h_msl_s_p,850,500,interp_log),2)
        et_dict['lapse_rate_700_500mb_c_km']=mpi._round_quantity_magnitude(get_lr_p(p_s_p,T_s_p,h_msl_s_p,700,500,interp_log),2)

        fzl_h, wbz_h = np.nan*self.units.m, np.nan*self.units.m
        if hasattr(self.mpcalc,'find_intersections'):
            zero_C_array = np.zeros_like(p.magnitude) * self.units.degC 

            try: 
                logger.debug(f"FZL Check: p type: {type(p)}, p shape: {getattr(p,'shape', 'N/A')}, p units: {getattr(p,'units','N/A')}, p[:3]: {p[:3] if hasattr(p,'__len__') and len(p)>2 else p}")
                logger.debug(f"FZL Check: T type: {type(T)}, T shape: {getattr(T,'shape', 'N/A')}, T units: {getattr(T,'units','N/A')}, T[:3]: {T[:3] if hasattr(T,'__len__') and len(T)>2 else T}")
                fzl_pc_tuple = self.mpcalc.find_intersections(p,T,zero_C_array,direction='decreasing')
                fzl_pc = fzl_pc_tuple[0] 
                if fzl_pc.size>0 and pd.notna(fzl_pc[0].m): fzl_h=mpi.interpolate_profile_value(fzl_pc[0],p,h_agl,interp_log)
            except Exception as e: logger.warning(f"FZL find_intersections: {e}", exc_info=True)
            try: 
                if hasattr(self.mpcalc,'wet_bulb_temperature'):
                    Tw = self.mpcalc.wet_bulb_temperature(p,T,Td)
                    logger.debug(f"WBZ Check: Tw type: {type(Tw)}, Tw shape: {getattr(Tw,'shape', 'N/A')}, Tw units: {getattr(Tw,'units','N/A')}, Tw[:3]: {Tw[:3] if hasattr(Tw,'__len__') and len(Tw)>2 else Tw}")
                    if hasattr(Tw,'size') and Tw.size>=2:
                        zero_C_array_for_Tw = np.zeros_like(Tw.magnitude) * self.units.degC if Tw.shape != p.shape else zero_C_array
                        wbz_pc_tuple = self.mpcalc.find_intersections(p,Tw,zero_C_array_for_Tw,direction='decreasing')
                        wbz_pc = wbz_pc_tuple[0] 
                        if wbz_pc.size>0 and pd.notna(wbz_pc[0].m): wbz_h=mpi.interpolate_profile_value(wbz_pc[0],p,h_agl,interp_log)
            except Exception as e: logger.warning(f"WBZ find_intersections: {e}", exc_info=True)
        et_dict['freezing_level_h_m_agl']=mpi._round_quantity_magnitude(fzl_h,0,'m')
        et_dict['wet_bulb_zero_h_m_agl']=mpi._round_quantity_magnitude(wbz_h,0,'m')

        if hasattr(self.mpcalc,'ccl'):
            try: _,_,cT=self.mpcalc.ccl(p,T,Td); et_dict['convective_temp_c']=mpi._round_quantity_magnitude(cT,1,'degC')
            except Exception as e: logger.warning(f"CCL ConvT: {e}", exc_info=True)
        if hasattr(self.mpcalc,'equivalent_potential_temperature'):
            try: et_dict['sfc_theta_e_k']=mpi._round_quantity_magnitude(self.mpcalc.equivalent_potential_temperature(sfc_p,sfc_T,sfc_Td),1,'K')
            except Exception as e: logger.warning(f"SFC ThetaE: {e}", exc_info=True)
        try: et_dict['sfc_dewpoint_depression_c']=mpi._round_quantity_magnitude(sfc_T-sfc_Td,1,'delta_degC')
        except Exception as e: logger.warning(f"SFC DPD: {e}", exc_info=True)

    def _calculate_composite_indices(self, sounding: Sounding):
        prof_data = self._get_prof_data(sounding)
        ci_dict = sounding.composite_indices
        p,h_agl = prof_data['p'], prof_data['hght_agl'] 
        logger.debug(f"Calculating composites for {sounding.metadata.get('STID','N/A')}")

        sfc_cape=sounding.parcel_thermodynamics.get('sfc_cape_jkg',np.nan)
        mlc100=sounding.parcel_thermodynamics.get('mlcape_100hpa_jkg',np.nan)
        mucape=sounding.parcel_thermodynamics.get('mucape_jkg',np.nan)
        dcape=sounding.parcel_thermodynamics.get('dcape_jkg',np.nan)
        
        mlp,mlT,mlTd = (np.nan*self.units.hPa, np.nan*self.units.degC, np.nan*self.units.degC)
        if hasattr(self.mpcalc,'mixed_parcel'):
            try: mlp,mlT,mlTd = self.mpcalc.mixed_parcel(p,prof_data['T'],prof_data['Td'],depth=app_config.ML_DEPTH_HPA*self.units.hPa)
            except Exception: pass
        mllcl_p, stp_mllcl_h = np.nan*self.units.hPa, np.nan
        interp_log_comp = 'log' if app_config.METPY_LOG_INTERPOLATE_1D_AVAILABLE else 'numpy_log'
        if hasattr(self.mpcalc,'lcl') and all(pd.notna(getattr(v,'m',v)) for v in [mlp,mlT,mlTd]):
            try:
                mllcl_p,_ = self.mpcalc.lcl(mlp,mlT,mlTd)
                mllcl_h_val = mpi.interpolate_profile_value(mllcl_p,p,h_agl,interp_log_comp)
                if hasattr(mllcl_h_val,'m') and pd.notna(mllcl_h_val.m): stp_mllcl_h = mllcl_h_val.to(self.units.m).m
            except Exception: pass
        
        srh01=sounding.detailed_kinematics.get('srh_0_1km_m2s2_eff_sm',np.nan)
        srh03=sounding.detailed_kinematics.get('srh_0_3km_m2s2_eff_sm',np.nan)
        shr06kts=sounding.detailed_kinematics.get('bulk_shear_0_6km_kts',np.nan)
        lr75=sounding.environmental_thermodynamics.get('lapse_rate_700_500mb_c_km',np.nan)
        pwat=sounding.environmental_thermodynamics.get('precipitable_water_mm',np.nan)
        t500=sounding.key_levels_data.get("500mb",{}).get('t_c',np.nan)

        shr06ms = (shr06kts*self.units.knots).to('m/s').m if pd.notna(shr06kts) else np.nan
        if all(pd.notna(x) for x in [mlc100,stp_mllcl_h,srh01,shr06ms]):
            t1=mlc100/1500.0; t2=(2000.0-stp_mllcl_h)/1000.0 if stp_mllcl_h<2000.0 else 0.0
            if stp_mllcl_h<1000.0:t2=1.0
            t3=srh01/150.0; t4=shr06ms/20.0
            if shr06ms<12.5:t4=0.0
            if shr06ms>30.0:t4=1.5
            ci_dict['stp_fixed_ml']=mpi._round_quantity_magnitude(max(0.0,t1*t2*t3*t4),2)
        
        ehi_d=160000.0
        if pd.notna(sfc_cape) and pd.notna(srh01): ci_dict['ehi_sfc_0_1km']=mpi._round_quantity_magnitude(max(0.0,(sfc_cape*srh01)/ehi_d),2)
        if pd.notna(sfc_cape) and pd.notna(srh03): ci_dict['ehi_sfc_0_3km']=mpi._round_quantity_magnitude(max(0.0,(sfc_cape*srh03)/ehi_d),2)
        if pd.notna(mlc100) and pd.notna(srh01): ci_dict['ehi_ml100_0_1km']=mpi._round_quantity_magnitude(max(0.0,(mlc100*srh01)/ehi_d),2)
        if pd.notna(mlc100) and pd.notna(srh03): ci_dict['ehi_ml100_0_3km']=mpi._round_quantity_magnitude(max(0.0,(mlc100*srh03)/ehi_d),2)
        ci_dict['ehi_effective_mu']=np.nan; ci_dict['scp_effective_mu']=np.nan

        mup_p = sounding.metadata.get('_intermediate_mup_p',np.nan*self.units.hPa)
        mup_Td = sounding.metadata.get('_intermediate_mup_Td',np.nan*self.units.degC)
        mr_gkg = np.nan
        if hasattr(self.mpcalc,'mixing_ratio') and all(hasattr(q,'m')and pd.notna(q.m) for q in [mup_p,mup_Td]):
            try:
                e=self.mpcalc.saturation_vapor_pressure(mup_Td); mr=self.mpcalc.mixing_ratio(e,mup_p)
                if hasattr(mr,'m') and pd.notna(mr.m): mr_gkg=mr.to_base_units().m*1000.0
            except Exception as ex: logger.warning(f"SHIP MU MR: {ex}", exc_info=True)
        logger.debug(f"SHIP comps: MUCAPE={mucape:.1f}, MR={mr_gkg:.2f}, LR75={lr75:.2f}, T500={t500:.1f}, S06={shr06kts:.1f}")
        if all(pd.notna(v) for v in [mucape,mr_gkg,lr75,t500,shr06kts]):
            s06m = (shr06kts*self.units.knots).to('m/s').m
            nmc=mucape/2000.0; nmr=mr_gkg/12.0; nlr=abs(lr75)/7.0 
            t5f=max(0.0,-t500); nt5=t5f/15.0; ns6=s06m/20.0
            ship=nmc*nmr*nlr*nt5*ns6*5.0
            logger.debug(f"SHIP norms: CAPE={nmc:.2f} MR={nmr:.2f} LR={nlr:.2f} T500F={t5f:.1f} T500N={nt5:.2f} S06N={ns6:.2f} -> SHIP={ship:.2f}")
            ci_dict['ship_calculated_v1']=mpi._round_quantity_magnitude(max(0.0,ship),2)

        if all(pd.notna(v) for v in [mucape,shr06kts,dcape,pwat]):
            s06m_d=(shr06kts*self.units.knots).to('m/s').m; pwat_i=(pwat*self.units.mm).to('in').m
            tmu=mucape/2000.0; ts6=s06m_d/20.0; tdc=dcape/900.0; tpw=pwat_i/1.5
            ci_dict['dcp_calculated']=mpi._round_quantity_magnitude(max(0.0,tmu*ts6*tdc*tpw),2)
        ci_dict['sweat_index_bufkit_summary']=sounding.summary_params.get('SWET',np.nan)

    def _calculate_inversion_characteristics(self, sounding: Sounding):
        prof_data = self._get_prof_data(sounding)
        inv_dict = sounding.inversion_characteristics
        p,T,Td,h_agl = prof_data['p'],prof_data['T'],prof_data['Td'],prof_data['hght_agl']
        logger.debug(f"Calculating inversions for {sounding.metadata.get('STID','N/A')}")

        sfc_cin_val = sounding.parcel_thermodynamics.get('sfc_cin_jkg',0.0) 
        sfc_T_inv,sfc_Td_inv = T[0],Td[0]
        sfc_prof_inv, sfc_lfc_p_inv = np.nan*self.units.degC, np.nan*self.units.hPa
        if hasattr(self.mpcalc,'parcel_profile'):
            try:
                sfc_prof_inv=self.mpcalc.parcel_profile(p,sfc_T_inv,sfc_Td_inv).to(self.units.degC)
                if hasattr(self.mpcalc,'lfc'):sfc_lfc_p_inv,_=self.mpcalc.lfc(p,T,Td,parcel_temperature_profile=sfc_prof_inv)
            except Exception: pass
        if hasattr(sfc_prof_inv,'units') and sfc_prof_inv.size==p.size and \
           pd.notna(sfc_cin_val) and sfc_cin_val<-1.0 and hasattr(sfc_lfc_p_inv,'m') and pd.notna(sfc_lfc_p_inv.m):
            try:
                mask=(p>=sfc_lfc_p_inv)&(sfc_prof_inv<T)
                if np.any(mask):
                    Te_cap,Tp_cap,p_cap,h_cap = T[mask],sfc_prof_inv[mask],p[mask],h_agl[mask]
                    if Te_cap.size>0: 
                        cap_s=np.max(Te_cap-Tp_cap)
                        inv_dict['sfc_cap_strength_c']=mpi._round_quantity_magnitude(cap_s,1,'delta_degC')
                        inv_dict['sfc_cap_base_p_hpa']=mpi._round_quantity_magnitude(p_cap[0],1,'hPa')
                        inv_dict['sfc_cap_top_p_hpa']=mpi._round_quantity_magnitude(p_cap[-1],1,'hPa')
                        inv_dict['sfc_cap_base_h_m_agl']=mpi._round_quantity_magnitude(h_cap[0],0,'m')
                        inv_dict['sfc_cap_top_h_m_agl']=mpi._round_quantity_magnitude(h_cap[-1],0,'m')
            except Exception as e: logger.warning(f"SFC Cap: {e}", exc_info=True)

    def _calculate_key_levels_snapshot(self, sounding: Sounding):
        prof_data = self._get_prof_data(sounding)
        kl_dict = sounding.key_levels_data
        p,T,Td,h_agl,u,v = prof_data['p'],prof_data['T'],prof_data['Td'],prof_data['hght_agl'],prof_data['u'],prof_data['v']
        logger.debug(f"Calculating key levels for {sounding.metadata.get('STID','N/A')}")
        interp_log_kl = 'log' if app_config.METPY_LOG_INTERPOLATE_1D_AVAILABLE else 'numpy_log'

        sfc_u_val, sfc_v_val = (u[0] if u.size > 0 else np.nan * self.units.knots), \
                               (v[0] if v.size > 0 else np.nan * self.units.knots)
        
        sfc_wind_spd, sfc_wind_dir = np.nan, np.nan
        if hasattr(self.mpcalc,'wind_speed') and hasattr(sfc_u_val,'m') and pd.notna(sfc_u_val.m):
            try: sfc_wind_spd = mpi._round_quantity_magnitude(self.mpcalc.wind_speed(sfc_u_val,sfc_v_val),1,'knots')
            except Exception: pass
        if hasattr(self.mpcalc,'wind_direction') and hasattr(sfc_u_val,'m') and pd.notna(sfc_u_val.m):
            try: sfc_wind_dir = mpi._round_quantity_magnitude(self.mpcalc.wind_direction(sfc_u_val,sfc_v_val),0,'degree')
            except Exception: pass

        kl_dict["sfc"] = {
            "p_hpa": mpi._round_quantity_magnitude(p[0],1,'hPa') if p.size > 0 else np.nan,
            "h_m_agl": mpi._round_quantity_magnitude(h_agl[0],0,'m') if h_agl.size > 0 else np.nan,
            "t_c": mpi._round_quantity_magnitude(T[0],1,'degC') if T.size > 0 else np.nan,
            "td_c": mpi._round_quantity_magnitude(Td[0],1,'degC') if Td.size > 0 else np.nan,
            "u_kts": mpi._round_quantity_magnitude(sfc_u_val,1,'knots'),
            "v_kts": mpi._round_quantity_magnitude(sfc_v_val,1,'knots'),
            "wind_spd_kts": sfc_wind_spd,
            "wind_dir_deg": sfc_wind_dir
        }
        
        sort_p_idx = np.argsort(p.magnitude); p_s,T_s,Td_s,h_s,u_s,v_s = (arr[sort_p_idx] for arr in [p,T,Td,h_agl,u,v])
        min_p_val,max_p_val = (p_s[0].m if p_s.size>0 else np.nan), (p_s[-1].m if p_s.size>0 else np.nan)
        targets = [pl for pl in app_config.STANDARD_P_LEVELS_HPA if pd.notna(min_p_val) and pd.notna(max_p_val) and min_p_val<=pl<=max_p_val]

        for p_hpa_val in targets:
            pq_target = p_hpa_val*self.units.hPa; key_str=f"{int(p_hpa_val)}mb"; current_level_dat={"p_hpa":p_hpa_val}
            try:
                current_level_dat["h_m_agl"]=mpi._round_quantity_magnitude(mpi.interpolate_profile_value(pq_target,p_s,h_s,interp_log_kl),0,'m')
                current_level_dat["t_c"]=mpi._round_quantity_magnitude(mpi.interpolate_profile_value(pq_target,p_s,T_s,interp_log_kl),1,'degC')
                current_level_dat["td_c"]=mpi._round_quantity_magnitude(mpi.interpolate_profile_value(pq_target,p_s,Td_s,interp_log_kl),1,'degC')
                iu_kl_val=mpi.interpolate_profile_value(pq_target,p_s,u_s,interp_log_kl)
                iv_kl_val=mpi.interpolate_profile_value(pq_target,p_s,v_s,interp_log_kl)
                current_level_dat["u_kts"]=mpi._round_quantity_magnitude(iu_kl_val,1,'knots')
                current_level_dat["v_kts"]=mpi._round_quantity_magnitude(iv_kl_val,1,'knots')
                
                kl_wind_spd_val, kl_wind_dir_val = np.nan, np.nan
                if hasattr(self.mpcalc,'wind_speed') and hasattr(iu_kl_val,'m') and pd.notna(iu_kl_val.m): 
                    try: kl_wind_spd_val = mpi._round_quantity_magnitude(self.mpcalc.wind_speed(iu_kl_val,iv_kl_val),1,'knots')
                    except Exception: pass
                if hasattr(self.mpcalc,'wind_direction') and hasattr(iu_kl_val,'m') and pd.notna(iu_kl_val.m):
                    try: kl_wind_dir_val = mpi._round_quantity_magnitude(self.mpcalc.wind_direction(iu_kl_val,iv_kl_val),0,'degree')
                    except Exception: pass
                current_level_dat["wind_spd_kts"]=kl_wind_spd_val
                current_level_dat["wind_dir_deg"]=kl_wind_dir_val
                kl_dict[key_str]=current_level_dat
            except Exception as e_kl: 
                default_kl_data = {"p_hpa":p_hpa_val, "h_m_agl":np.nan, "t_c":np.nan, "td_c":np.nan, 
                                   "u_kts":np.nan, "v_kts":np.nan, "wind_spd_kts":np.nan, "wind_dir_deg":np.nan}
                kl_dict[key_str] = default_kl_data
                logger.warning(f"KeyLvl {key_str}: {e_kl}", exc_info=True)