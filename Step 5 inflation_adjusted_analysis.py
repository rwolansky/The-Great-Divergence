import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import linregress, normaltest, jarque_bera
import warnings
warnings.filterwarnings('ignore')

# ── File paths ────────────────────────────────────────────────────────────────
CPT_FILE      = r"Z:\homes\Rachel Wolansky\The Great Divergence\Results\payment_statistics_inflation_adjusted.txt"
DRG_FILE      = r"Z:\homes\Rachel Wolansky\The Great Divergence\Results\drg_payment_statistics_inflation_adjusted.txt"
RATIOS_CSV    = r"Z:\homes\Rachel Wolansky\The Great Divergence\Results\drg_cpt_ratios_data.csv"   # from Step 4
RESULTS_DIR   = r"Z:\homes\Rachel Wolansky\The Great Divergence\Results"

STATS_CSV     = os.path.join(RESULTS_DIR, "time_series_analysis_inflation_adjusted_results.csv")

CHART_FILES = {
    "fig1": os.path.join(RESULTS_DIR, "fig1_avg_ratio_by_procedure_type.png"),
    "fig2": os.path.join(RESULTS_DIR, "fig2_top5_procedure_ratios.png"),
    "fig3": os.path.join(RESULTS_DIR, "fig3_drg_cpt_payment_index.png"),
}

YEARS = np.array([2019, 2020, 2021, 2022, 2023])

# CPT codes for the 5 abstract procedures used across Figs 2, 3, 4
ABSTRACT_PROCS = {
    '34701': ('EVAR (CPT 34701)',                   'VASC',  268),
    '33945': ('Heart Transplant (CPT 33945)',        'TX3',   1),
    '47135': ('Liver Transplant (CPT 47135)',        'TX2',   5),
    '47562': ('Cholecystectomy (CPT 47562)',         'CHOL',  417),
    '44120': ('Small Bowel Resection (CPT 44120)',   'SB1',   329),
}
ABSTRACT_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']


# ═══════════════════════════════════════════════════════════════════════════════
# PARSERS (for raw payment data needed by Figs 5 & 6)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_cpt_summary(filepath):
    """Return {cpt_int: [val_2019..val_2023]} from inflation-adjusted CPT file."""
    with open(filepath, 'r', encoding='utf-8') as fh:
        content = fh.read()
    start = content.find("SUMMARY TABLE - NON-FACILITY MEAN PAYMENTS BY YEAR")
    if start == -1:
        raise ValueError("CPT summary table not found.")
    data, in_data = {}, False
    for line in content[start:].split('\n'):
        s = line.strip()
        if '--' in s:
            in_data = True
            continue
        if not in_data or not s or s.startswith('='):
            continue
        if len(s) >= 5 and s[:5].isdigit():
            cpt = int(s[:5])
            vals = [float(m.replace(',', ''))
                    for m in re.findall(r'\$\s*([\d,]+\.?\d*)', s[5:])]
            if len(vals) == 5:
                data[cpt] = vals
    return data


def parse_drg_summary(filepath):
    """Return {drg_int: [val_2019..val_2023]} from inflation-adjusted DRG file."""
    with open(filepath, 'r', encoding='utf-8') as fh:
        content = fh.read()
    start = content.find("SUMMARY TABLE - MEAN PAYMENTS BY DRG AND YEAR")
    if start == -1:
        raise ValueError("DRG summary table not found.")
    data, in_data = {}, False
    for line in content[start:].split('\n'):
        s = line.strip()
        if '--' in s:
            in_data = True
            continue
        if not in_data or not s or s.startswith('='):
            continue
        parts = s.split()
        if not parts or not parts[0].isdigit():
            continue
        drg = int(parts[0])
        na_count = s.count('N/A')
        dollar_vals = [float(m.replace(',', ''))
                       for m in re.findall(r'\$\s*([\d,]+(?:\.\d+)?)', s)]
        if na_count == 5:
            vals = [np.nan] * 5
        elif na_count > 0:
            vals = dollar_vals + [np.nan] * (5 - len(dollar_vals))
        else:
            vals = dollar_vals[-5:] if len(dollar_vals) >= 5 else dollar_vals
        if len(vals) == 5:
            data[drg] = vals
        elif 0 < len(vals) < 5:
            data[drg] = [np.nan] * (5 - len(vals)) + vals
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class TimeSeriesTrendAnalyzer:
    """
    Time series analysis for DRG/CPT payment ratio trends.
    Uses inflation-adjusted payment data (2023 dollars).
    """

    def __init__(self):
        self.years   = YEARS
        self.n_years = len(self.years)
        self.results = []
        # Corrected p-values stored after run_analysis() for use in figures
        self.corrected_p_max_reg = []
        self.corrected_p_min_reg = []
        self.corrected_p_max_mk  = []
        self.corrected_p_min_mk  = []

    def parse_cpt_file(self, filepath):
        return parse_cpt_summary(filepath)

    def parse_drg_file(self, filepath):
        return parse_drg_summary(filepath)

    def load_data(self, cpt_filepath=None, drg_filepath=None):
        if cpt_filepath is None:
            cpt_filepath = CPT_FILE
        if drg_filepath is None:
            drg_filepath = DRG_FILE

        print(f"Loading inflation-adjusted payment data...")
        print(f"CPT file: {cpt_filepath}")
        print(f"DRG file: {drg_filepath}")
        print("Note: All values are in 2023 dollars\n")

        if not os.path.exists(cpt_filepath):
            print(f"Error: CPT file not found: {cpt_filepath}")
            return []
        if not os.path.exists(drg_filepath):
            print(f"Error: DRG file not found: {drg_filepath}")
            return []

        cpt_data = self.parse_cpt_file(cpt_filepath)
        drg_data = self.parse_drg_file(drg_filepath)
        print(f"Loaded {len(cpt_data)} CPT codes")
        print(f"Loaded {len(drg_data)} DRG codes")

        if not cpt_data or not drg_data:
            print("Error: Could not load payment data")
            return []

        crosswalk = {
            'APPY':  {'cpt': 44970, 'drg_max': 339,  'drg_min': 343},
            'CHOL':  {'cpt': 47562, 'drg_max': 417,  'drg_min': 419},
            'COLO1': {'cpt': 44140, 'drg_max': 329,  'drg_min': 331},
            'COLO2': {'cpt': 44204, 'drg_max': 329,  'drg_min': 331},
            'COLO3': {'cpt': 44206, 'drg_max': 329,  'drg_min': 331},
            'COLO4': {'cpt': 44143, 'drg_max': 329,  'drg_min': 331},
            'SB1':   {'cpt': 44120, 'drg_max': 329,  'drg_min': 331},
            'SB2':   {'cpt': 44005, 'drg_max': 388,  'drg_min': 390},
            'SB3':   {'cpt': 44180, 'drg_max': 388,  'drg_min': 390},
            'GAST1': {'cpt': 43644, 'drg_max': 619,  'drg_min': 621},
            'GAST2': {'cpt': 43775, 'drg_max': 619,  'drg_min': 621},
            'HEP':   {'cpt': 47120, 'drg_max': 405,  'drg_min': 407},
            'TX1':   {'cpt': 50360, 'drg_max': 650,  'drg_min': 652},
            'TX2':   {'cpt': 47135, 'drg_max': 5,    'drg_min': 6},
            'TX3':   {'cpt': 33945, 'drg_max': 1,    'drg_min': 2},
            'VASC':  {'cpt': 34701, 'drg_max': 268,  'drg_min': 269},
        }

        ratio_data = []
        for proc_name, mapping in crosswalk.items():
            cpt_code    = mapping['cpt']
            drg_max_code = mapping['drg_max']
            drg_min_code = mapping['drg_min']

            if cpt_code not in cpt_data:
                print(f"Warning: CPT {cpt_code} not found for {proc_name}")
                continue

            cpt_payments     = np.array(cpt_data[cpt_code])
            drg_max_payments = np.array(drg_data.get(drg_max_code, [np.nan]*5))
            drg_min_payments = np.array(drg_data.get(drg_min_code, [np.nan]*5))

            if drg_max_code not in drg_data:
                print(f"Warning: DRG {drg_max_code} not found for {proc_name}")
            if drg_min_code not in drg_data:
                print(f"Warning: DRG {drg_min_code} not found for {proc_name}")

            with np.errstate(divide='ignore', invalid='ignore'):
                max_ratios = np.where(cpt_payments != 0, drg_max_payments / cpt_payments, np.nan)
                min_ratios = np.where(cpt_payments != 0, drg_min_payments / cpt_payments, np.nan)
            max_ratios = np.where(np.isfinite(max_ratios), max_ratios, np.nan)
            min_ratios = np.where(np.isfinite(min_ratios), min_ratios, np.nan)

            ratio_data.append({
                'procedure':        proc_name,
                'cpt_code':         cpt_code,
                'drg_max_code':     drg_max_code,
                'drg_min_code':     drg_min_code,
                'max_ratios':       max_ratios,
                'min_ratios':       min_ratios,
                'cpt_payments':     cpt_payments,
                'drg_max_payments': drg_max_payments,
                'drg_min_payments': drg_min_payments,
            })

        print(f"Successfully processed {len(ratio_data)} procedure mappings")
        return ratio_data

    def mann_kendall_test(self, data):
        n = len(data)
        if n < 3:
            return {'insufficient_data': True, 'n': n}
        valid_data = data[~np.isnan(data)]
        n = len(valid_data)
        if n < 3:
            return {'insufficient_data': True, 'n': n}
        S = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if valid_data[j] > valid_data[i]:   S += 1
                elif valid_data[j] < valid_data[i]: S -= 1
        var_S = n * (n - 1) * (2 * n + 5) / 18
        Z = (S - 1) / np.sqrt(var_S) if S > 0 else ((S + 1) / np.sqrt(var_S) if S < 0 else 0)
        p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
        return {
            'S': S, 'Z': Z, 'p_value': p_value, 'n': n,
            'trend': 'increasing' if S > 0 else ('decreasing' if S < 0 else 'no trend'),
            'significant': p_value < 0.05,
        }

    def sens_slope_estimator(self, years, data):
        valid_idx  = ~np.isnan(data)
        valid_years = years[valid_idx]
        valid_data  = data[valid_idx]
        if len(valid_data) < 3:
            return {'insufficient_data': True}
        slopes = []
        n = len(valid_data)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if valid_years[j] != valid_years[i]:
                    slopes.append((valid_data[j] - valid_data[i]) / (valid_years[j] - valid_years[i]))
        if not slopes:
            return {'insufficient_data': True}
        return {
            'sens_slope':  np.median(slopes),
            'n_slopes':    len(slopes),
            'slope_range': (np.min(slopes), np.max(slopes)),
        }

    def linear_regression_analysis(self, years, data):
        valid_idx  = ~np.isnan(data)
        valid_years = years[valid_idx]
        valid_data  = data[valid_idx]
        if len(valid_data) < 3:
            return {'insufficient_data': True, 'n': len(valid_data)}
        slope, intercept, r_value, p_value, std_err = linregress(valid_years, valid_data)
        n  = len(valid_data)
        df = n - 2
        t_critical = stats.t.ppf(0.975, df)
        ci_lower   = slope - t_critical * std_err
        ci_upper   = slope + t_critical * std_err
        predicted  = slope * valid_years + intercept
        residuals  = valid_data - predicted
        dw_stat    = np.sum(np.diff(residuals)**2) / np.sum(residuals**2) if n > 2 else np.nan
        shapiro_stat, shapiro_p = (stats.shapiro(residuals) if n >= 8 else (np.nan, np.nan))
        percent_change = ((valid_data[-1] - valid_data[0]) / valid_data[0]) * 100 if len(valid_data) >= 2 else np.nan
        return {
            'n': n, 'slope': slope, 'intercept': intercept,
            'r_value': r_value, 'r_squared': r_value**2,
            'p_value': p_value, 'std_err': std_err,
            'ci_lower': ci_lower, 'ci_upper': ci_upper,
            'percent_change': percent_change,
            'durbin_watson': dw_stat,
            'shapiro_stat': shapiro_stat, 'shapiro_p': shapiro_p,
            'residuals': residuals,
            'significant': p_value < 0.05,
        }

    def analyze_procedure(self, proc_data):
        procedure = proc_data['procedure']
        results = {
            'procedure':        procedure,
            'cpt_code':         proc_data['cpt_code'],
            'drg_max_code':     proc_data['drg_max_code'],
            'drg_min_code':     proc_data['drg_min_code'],
            'cpt_payments':     proc_data['cpt_payments'],
            'drg_max_payments': proc_data['drg_max_payments'],
        }
        for ratio_type, key in [('max_ratios', 'max'), ('min_ratios', 'min')]:
            ratios = proc_data[ratio_type]
            if not np.all(np.isnan(ratios)):
                results[f'{key}_ratio_regression']   = self.linear_regression_analysis(self.years, ratios)
                results[f'{key}_ratio_mann_kendall']  = self.mann_kendall_test(ratios)
                results[f'{key}_ratio_sens_slope']    = self.sens_slope_estimator(self.years, ratios)
            else:
                results[f'{key}_ratio_regression']  = {'insufficient_data': True}
                results[f'{key}_ratio_mann_kendall'] = {'insufficient_data': True}
                results[f'{key}_ratio_sens_slope']   = {'insufficient_data': True}
        return results

    def multiple_testing_correction(self, p_values):
        valid_indices = [(i, p) for i, p in enumerate(p_values) if not np.isnan(p)]
        if not valid_indices:
            return p_values
        orig_indices, valid_p_values = zip(*valid_indices)
        n = len(valid_p_values)
        sorted_pairs = sorted(zip(valid_p_values, orig_indices), key=lambda x: x[0])
        corrected_p  = list(p_values)
        for rank, (p_val, orig_idx) in enumerate(sorted_pairs, 1):
            corrected_p[orig_idx] = min(p_val * n / rank, 1.0)
        for i in range(len(sorted_pairs) - 1, 0, -1):
            _, curr_idx = sorted_pairs[i]
            _, prev_idx = sorted_pairs[i - 1]
            corrected_p[prev_idx] = min(corrected_p[prev_idx], corrected_p[curr_idx])
        return corrected_p

    def run_analysis(self, cpt_filepath=None, drg_filepath=None):
        print("=== TIME SERIES ANALYSIS FOR DRG/CPT PAYMENT RATIOS ===")
        print("=== USING INFLATION-ADJUSTED VALUES (2023 DOLLARS) ===")
        print("=" * 80)

        ratio_data = self.load_data(cpt_filepath, drg_filepath)
        if not ratio_data:
            print("No data available for analysis")
            return [], []

        all_results  = []
        pv_max_reg, pv_min_reg, pv_max_mk, pv_min_mk = [], [], [], []

        for proc_data in ratio_data:
            result = self.analyze_procedure(proc_data)
            all_results.append(result)
            pv_max_reg.append(result['max_ratio_regression'].get('p_value', np.nan)
                               if not result['max_ratio_regression'].get('insufficient_data') else np.nan)
            pv_min_reg.append(result['min_ratio_regression'].get('p_value', np.nan)
                               if not result['min_ratio_regression'].get('insufficient_data') else np.nan)
            pv_max_mk.append(result['max_ratio_mann_kendall'].get('p_value', np.nan)
                              if not result['max_ratio_mann_kendall'].get('insufficient_data') else np.nan)
            pv_min_mk.append(result['min_ratio_mann_kendall'].get('p_value', np.nan)
                              if not result['min_ratio_mann_kendall'].get('insufficient_data') else np.nan)

        self.corrected_p_max_reg = self.multiple_testing_correction(pv_max_reg)
        self.corrected_p_min_reg = self.multiple_testing_correction(pv_min_reg)
        self.corrected_p_max_mk  = self.multiple_testing_correction(pv_max_mk)
        self.corrected_p_min_mk  = self.multiple_testing_correction(pv_min_mk)

        self.generate_report(all_results)
        return all_results, ratio_data

    def generate_report(self, results):
        print("\n1. LINEAR REGRESSION ANALYSIS (OLS) - INFLATION ADJUSTED")
        print("=" * 60)
        print(f"{'Procedure':<12} {'Type':<4} {'n':<2} {'Slope':<8} {'95% CI':<20} "
              f"{'R²':<6} {'p-val':<7} {'Corrected p':<11} {'Sig':<4}")
        print("-" * 85)
        for i, result in enumerate(results):
            proc = result['procedure']
            for ratio_type, cp_list in [('max', self.corrected_p_max_reg),
                                         ('min', self.corrected_p_min_reg)]:
                reg = result[f'{ratio_type}_ratio_regression']
                if not reg.get('insufficient_data'):
                    cp  = cp_list[i]
                    sig = "YES" if not np.isnan(cp) and cp < 0.05 else "NO"
                    print(f"{proc:<12} {ratio_type.capitalize():<4} {reg['n']:<2} {reg['slope']:<8.4f} "
                          f"[{reg['ci_lower']:.4f}, {reg['ci_upper']:.4f}]    "
                          f"{reg['r_squared']:<6.3f} {reg['p_value']:<7.3f} {cp:<11.3f} {sig:<4}")

        print("\n2. MANN-KENDALL TREND TEST (NON-PARAMETRIC) - INFLATION ADJUSTED")
        print("=" * 60)
        print(f"{'Procedure':<12} {'Type':<4} {'n':<2} {'S':<6} {'Z':<8} "
              f"{'p-val':<7} {'Corrected p':<11} {'Trend':<11} {'Sig':<4}")
        print("-" * 80)
        for i, result in enumerate(results):
            proc = result['procedure']
            for ratio_type, cp_list in [('max', self.corrected_p_max_mk),
                                         ('min', self.corrected_p_min_mk)]:
                mk = result[f'{ratio_type}_ratio_mann_kendall']
                if not mk.get('insufficient_data'):
                    cp  = cp_list[i]
                    sig = "YES" if not np.isnan(cp) and cp < 0.05 else "NO"
                    print(f"{proc:<12} {ratio_type.capitalize():<4} {mk['n']:<2} {mk['S']:<6} "
                          f"{mk['Z']:<8.2f} {mk['p_value']:<7.3f} {cp:<11.3f} "
                          f"{mk['trend']:<11} {sig:<4}")

        print("\n3. SEN'S SLOPE ESTIMATES - INFLATION ADJUSTED")
        print("=" * 60)
        for result in results:
            proc = result['procedure']
            for ratio_type in ['max', 'min']:
                ss = result[f'{ratio_type}_ratio_sens_slope']
                if not ss.get('insufficient_data'):
                    print(f"{proc:<12} {ratio_type.capitalize():<4} "
                          f"{ss['sens_slope']:<10.3f} "
                          f"[{ss['slope_range'][0]:.3f}, {ss['slope_range'][1]:.3f}]")

        print("\n4. SUMMARY - BH FDR CORRECTION")
        print("=" * 60)
        all_cp = self.corrected_p_max_mk + self.corrected_p_min_mk
        sig    = sum(1 for p in all_cp if not np.isnan(p) and p < 0.05)
        total  = sum(1 for p in all_cp if not np.isnan(p))
        print(f"Significant trends (Mann-Kendall, BH corrected): {sig}/{total}")
        print(f"Method: Benjamini-Hochberg, alpha = 0.05")

        print("\n5. CPT AND DRG ABSOLUTE PAYMENT TRENDS (2019-2023, INFLATION-ADJUSTED)")
        print("=" * 80)
        print("Tracks real-dollar change in CPT facility fees and DRG-Max payments independently.")
        print(f"\n{'Procedure':<12} {'CPT 2019':>10} {'CPT 2023':>10} {'CPT %Chg':>9} "
              f"{'DRG 2019':>10} {'DRG 2023':>10} {'DRG %Chg':>9} {'OLS CPT Slope':>14}")
        print("-" * 90)

        all_cpt_pct_changes = []
        all_drg_pct_changes = []

        for result in results:
            proc         = result['procedure']
            cpt_payments = result.get('cpt_payments')
            drg_max      = result.get('drg_max_payments')

            if cpt_payments is None or drg_max is None:
                continue

            cpt_arr = np.array(cpt_payments, dtype=float)
            drg_arr = np.array(drg_max, dtype=float)

            # CPT % change 2019 -> 2023
            cpt_2019 = cpt_arr[0] if np.isfinite(cpt_arr[0]) else np.nan
            cpt_2023 = cpt_arr[-1] if np.isfinite(cpt_arr[-1]) else np.nan
            cpt_pct  = ((cpt_2023 - cpt_2019) / cpt_2019 * 100) if not np.isnan(cpt_2019) and cpt_2019 != 0 else np.nan

            # DRG % change 2019 -> 2023
            drg_2019 = drg_arr[0] if np.isfinite(drg_arr[0]) else np.nan
            drg_2023 = drg_arr[-1] if np.isfinite(drg_arr[-1]) else np.nan
            drg_pct  = ((drg_2023 - drg_2019) / drg_2019 * 100) if not np.isnan(drg_2019) and drg_2019 != 0 else np.nan

            # OLS slope on CPT payments
            mask = np.isfinite(cpt_arr)
            if mask.sum() >= 3:
                cpt_slope, *_ = linregress(self.years[mask], cpt_arr[mask])
                cpt_slope_pct = cpt_slope / cpt_2019 * 100 if not np.isnan(cpt_2019) else np.nan
            else:
                cpt_slope = np.nan
                cpt_slope_pct = np.nan

            if not np.isnan(cpt_pct): all_cpt_pct_changes.append(cpt_pct)
            if not np.isnan(drg_pct): all_drg_pct_changes.append(drg_pct)

            cpt_2019_s = f"${cpt_2019:>8,.0f}" if not np.isnan(cpt_2019) else "N/A"
            cpt_2023_s = f"${cpt_2023:>8,.0f}" if not np.isnan(cpt_2023) else "N/A"
            cpt_pct_s  = f"{cpt_pct:>+8.1f}%" if not np.isnan(cpt_pct) else "N/A"
            drg_2019_s = f"${drg_2019:>8,.0f}" if not np.isnan(drg_2019) else "N/A"
            drg_2023_s = f"${drg_2023:>8,.0f}" if not np.isnan(drg_2023) else "N/A"
            drg_pct_s  = f"{drg_pct:>+8.1f}%" if not np.isnan(drg_pct) else "N/A"
            slope_s    = f"{cpt_slope_pct:>+8.1f}%/yr" if not np.isnan(cpt_slope_pct) else "N/A"

            print(f"{proc:<12} {cpt_2019_s:>10} {cpt_2023_s:>10} {cpt_pct_s:>9} "
                  f"{drg_2019_s:>10} {drg_2023_s:>10} {drg_pct_s:>9} {slope_s:>14}")

        print("-" * 90)
        if all_cpt_pct_changes:
            print(f"{'MEAN':<12} {'':>10} {'':>10} {np.mean(all_cpt_pct_changes):>+8.1f}% "
                  f"{'':>10} {'':>10} {np.mean(all_drg_pct_changes):>+8.1f}%")
        print("\nNote: Values are inflation-adjusted (2023 dollars).")
        print("CPT % change reflects real erosion of physician payments.")
        print("DRG % change reflects real growth in hospital payments.")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def sig_label(bh_p):
    """Return a significance star string based on BH-corrected p-value."""
    if np.isnan(bh_p): return ""
    if bh_p < 0.05:    return "*"
    return ""


def get_stat_for_proc(proc_name, ratio_type, all_results, corrected_p_list):
    """
    Return (sens_slope, bh_p, bh_sig_label) for a given procedure and ratio type.
    ratio_type: 'max' or 'min'
    """
    for i, r in enumerate(all_results):
        if r['procedure'] == proc_name:
            ss  = r[f'{ratio_type}_ratio_sens_slope']
            reg = r[f'{ratio_type}_ratio_regression']
            bh_p = corrected_p_list[i]
            slope = ss.get('sens_slope', np.nan) if not ss.get('insufficient_data') else np.nan
            return slope, bh_p, sig_label(bh_p)
    return np.nan, np.nan, ""


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES 1–4  (from Step 4 data, annotated with Step 5 stats)
# ═══════════════════════════════════════════════════════════════════════════════

def save_fig1(results_df, all_results, analyzer):
    """Fig 1: Average ratio by procedure type with BH significance stars in legend."""
    proc_types = [t for t in results_df['Type'].unique() if pd.notna(t)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(proc_types)))

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, ptype in enumerate(proc_types):
        type_data  = results_df[results_df['Type'] == ptype]
        yearly_avg = type_data.groupby('Year')[['Ratio_Max', 'Ratio_Min']].mean()
        if yearly_avg.empty:
            continue

        # Collect BH-corrected p-values across all procedures of this type
        proc_codes = [r['procedure'] for r in all_results
                      if r['procedure'].startswith(ptype)]

        max_bh_ps = [analyzer.corrected_p_max_reg[j]
                     for j, r in enumerate(all_results)
                     if r['procedure'].startswith(ptype)
                     and not np.isnan(analyzer.corrected_p_max_reg[j])]
        min_bh_ps = [analyzer.corrected_p_min_reg[j]
                     for j, r in enumerate(all_results)
                     if r['procedure'].startswith(ptype)
                     and not np.isnan(analyzer.corrected_p_min_reg[j])]

        # Use minimum (most significant) BH p-value across procedures of this type
        best_max_p = np.nanmin(max_bh_ps) if max_bh_ps else np.nan
        best_min_p = np.nanmin(min_bh_ps) if min_bh_ps else np.nan

        max_label = f'{ptype} (Max){sig_label(best_max_p)}'
        min_label = f'{ptype} (Min){sig_label(best_min_p)}'

        ax.plot(yearly_avg.index, yearly_avg['Ratio_Max'],
                marker='o', color=colors[i], label=max_label)
        ax.plot(yearly_avg.index, yearly_avg['Ratio_Min'],
                marker='s', color=colors[i], linestyle='--', label=min_label)

    ax.set_xlabel('Year')
    ax.set_ylabel('Average DRG/CPT Ratio')
    ax.set_title('Average Payment Ratios by Procedure Type', fontsize=14, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=4, fontsize=7, frameon=True)
    fig.text(0.5, 0.01, '* p<0.05 (Benjamini-Hochberg corrected)',
             fontsize=6.5, ha='center', va='bottom', color='#444444')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([2019, 2020, 2021, 2022, 2023])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    fig.savefig(CHART_FILES["fig1"], dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {CHART_FILES['fig1']}")


def save_fig2(results_df, all_results, analyzer):
    """Fig 2: Top 5 procedures from abstract, annotated with BH-corrected Sen's slope."""
    # Keyed by CPT code: (display name, crosswalk procedure code for stat lookup)
    key_procs = {
        '34701': ('EVAR (CPT 34701)',                'VASC'),
        '33945': ('Heart Transplant (CPT 33945)',    'TX3'),
        '47135': ('Liver Transplant (CPT 47135)',    'TX2'),
        '47562': ('Cholecystectomy (CPT 47562)',     'CHOL'),
        '44120': ('Small Bowel Resection (CPT 44120)', 'SB1'),
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    # Ensure CPT column is string for matching
    results_df = results_df.copy()
    results_df['CPT'] = results_df['CPT'].astype(str).str.strip()

    for (cpt_code, (display_name, proc_code)), color in zip(key_procs.items(), ABSTRACT_COLORS):
        proc_data = results_df[results_df['CPT'] == cpt_code]
        if proc_data.empty:
            print(f"  Warning: CPT {cpt_code} not found in results_df")
            continue
        yearly = proc_data.groupby('Year')['Ratio_Max'].mean()
        if yearly.empty:
            continue
        slope, bh_p, sig = get_stat_for_proc(proc_code, 'max', all_results,
                                              analyzer.corrected_p_max_reg)
        label = (f"{display_name}  slope={slope:+.2f}/yr{sig}"
                 if not np.isnan(slope) else display_name)
        ax.plot(yearly.index, yearly.values, marker='o', color=color, linewidth=2, label=label)

    ax.set_xlabel('Year')
    ax.set_ylabel('DRG/CPT Ratio (Max DRG)')
    ax.set_title('Top 5 Procedures by DRG/CPT Ratio Increase', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, frameon=True)
    fig.text(0.5, 0.01, "Sen's slope shown; * p<0.05 Benjamini-Hochberg corrected",
             ha='center', va='bottom', fontsize=7, color='#444444')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([2019, 2020, 2021, 2022, 2023])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(CHART_FILES["fig2"], dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {CHART_FILES['fig2']}")


def save_fig3(cpt_data, drg_data, results_df, all_results, analyzer):
    """
    Fig 3: DRG and CPT payment index (2019=100) for the 5 abstract procedures.
    Shows both sides of the divergence — DRG rising, CPT flat/declining.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Ensure CPT column is string
    results_df = results_df.copy()
    results_df['CPT'] = results_df['CPT'].astype(str).str.strip()

    for (cpt_str, (display_name, proc_code, drg_code)), color in zip(ABSTRACT_PROCS.items(), ABSTRACT_COLORS):
        cpt_int = int(cpt_str)

        # CPT payment values from parsed summary
        cpt_vals = np.array(cpt_data.get(cpt_int, [np.nan]*5), dtype=float)
        # DRG payment values from parsed summary
        drg_vals = np.array(drg_data.get(drg_code, [np.nan]*5), dtype=float)

        cpt_baseline = cpt_vals[0]
        drg_baseline = drg_vals[0]

        if np.isnan(cpt_baseline) or cpt_baseline == 0:
            print(f"  Warning: CPT {cpt_str} missing baseline")
            continue
        if np.isnan(drg_baseline) or drg_baseline == 0:
            print(f"  Warning: DRG {drg_code} missing baseline")
            continue

        cpt_index = cpt_vals / cpt_baseline * 100
        drg_index = drg_vals / drg_baseline * 100

        # Plot DRG (solid) and CPT (dashed) with same color per procedure
        short = display_name.split('(')[0].strip()
        ax.plot(YEARS, drg_index, marker='o', color=color, linewidth=2,
                label=f"{short} — DRG")
        ax.plot(YEARS, cpt_index, marker='s', color=color, linewidth=2,
                linestyle='--', alpha=0.6, label=f"{short} — CPT")

    ax.axhline(y=100, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Payment Index (2019 = 100)', fontsize=11)
    ax.set_title('DRG vs CPT Payment Trends', fontsize=14, fontweight='bold')
    ax.set_xticks(YEARS)
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=5, frameon=True)
    fig.text(0.5, 0.01,
             'All values inflation-adjusted to 2023 dollars; index relative to 2019 baseline',
             fontsize=7, ha='center', va='bottom', color='#444444')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(CHART_FILES["fig3"], dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {CHART_FILES['fig3']}")




# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT STATS CSV
# ═══════════════════════════════════════════════════════════════════════════════

def export_stats_csv(all_results, analyzer):
    summary_data = []
    for i, result in enumerate(all_results):
        proc = result['procedure']
        for ratio_type, reg_key, mk_key, cp_reg, cp_mk in [
            ('Max', 'max_ratio_regression', 'max_ratio_mann_kendall',
             analyzer.corrected_p_max_reg, analyzer.corrected_p_max_mk),
            ('Min', 'min_ratio_regression', 'min_ratio_mann_kendall',
             analyzer.corrected_p_min_reg, analyzer.corrected_p_min_mk),
        ]:
            reg = result[reg_key]
            mk  = result[mk_key]
            ss  = result[reg_key.replace('regression', 'sens_slope')]
            if not reg.get('insufficient_data'):
                bh_reg = cp_reg[i]
                bh_mk  = cp_mk[i]
                summary_data.append({
                    'Procedure':           proc,
                    'Ratio_Type':          ratio_type,
                    'CPT_Code':            result['cpt_code'],
                    'DRG_Code':            result[f'drg_{ratio_type.lower()}_code'],
                    'Linear_Slope':        reg['slope'],
                    'Linear_R2':           reg['r_squared'],
                    'Linear_p_value':      reg['p_value'],
                    'Linear_BH_p_value':   bh_reg,
                    'Linear_Significant_BH': bool(not np.isnan(bh_reg) and bh_reg < 0.05),
                    'MK_Z':                mk.get('Z', np.nan),
                    'MK_p_value':          mk.get('p_value', np.nan),
                    'MK_BH_p_value':       bh_mk,
                    'MK_Significant_BH':   bool(not np.isnan(bh_mk) and bh_mk < 0.05),
                    'Sens_Slope':          ss.get('sens_slope', np.nan) if not ss.get('insufficient_data') else np.nan,
                    'Inflation_Adjusted':  True,
                })
    df = pd.DataFrame(summary_data)
    df.to_csv(STATS_CSV, index=False)
    print(f"\nStats CSV saved: {STATS_CSV}  ({len(df)} records)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        # ── Step 5a: Statistical analysis ────────────────────────────────────
        analyzer = TimeSeriesTrendAnalyzer()
        all_results, ratio_data = analyzer.run_analysis()

        if not all_results:
            print("No results — exiting.")
        else:
            # ── Step 5b: Export stats CSV ─────────────────────────────────────
            export_stats_csv(all_results, analyzer)

            # ── Step 5c: Load ratio data from Step 4 CSV ─────────────────────
            if not os.path.exists(RATIOS_CSV):
                print(f"\nWarning: {RATIOS_CSV} not found. "
                      "Run Step 4 first to generate Figs 1-4.")
                results_df = None
            else:
                results_df = pd.read_csv(RATIOS_CSV)
                print(f"\nLoaded ratio data from Step 4: {len(results_df)} rows")

            # ── Step 5d: Load raw payment data for Figs 3 & 4 ───────────────
            print("\nLoading raw payment summaries for Figs 3-4...")
            cpt_summary = parse_cpt_summary(CPT_FILE)
            drg_summary = parse_drg_summary(DRG_FILE)

            # ── Step 5e: Generate all figures ────────────────────────────────
            print("\nGenerating figures...")
            if results_df is not None and not results_df.empty:
                save_fig1(results_df, all_results, analyzer)
                save_fig2(results_df, all_results, analyzer)
            else:
                print("Skipping Figs 1-2 (no Step 4 ratio CSV found).")

            save_fig3(cpt_summary, drg_summary, results_df, all_results, analyzer)

            print("\n\nAll figures saved:")
            for key, path in CHART_FILES.items():
                print(f"  {key}: {path}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    input("\nPress Enter to close...")
