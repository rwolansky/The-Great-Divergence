import re
import os
import pandas as pd
import numpy as np
from collections import defaultdict

# ── File paths ────────────────────────────────────────────────────────────────
cpt_file       = r"Z:\homes\Rachel Wolansky\The Great Divergence\Results\payment_statistics_inflation_adjusted.txt"
drg_file       = r"Z:\homes\Rachel Wolansky\The Great Divergence\Results\drg_payment_statistics_inflation_adjusted.txt"
crosswalk_file = r"Z:\homes\Rachel Wolansky\The Great Divergence\Data In\CPT_DRG_ASA_Crosswalk.csv"
output_file    = r"Z:\homes\Rachel Wolansky\The Great Divergence\Results\drg_cpt_ratio_analysis_inflation_adjusted.txt"

# This CSV is passed to Step 5 for figure generation
ratios_csv     = r"Z:\homes\Rachel Wolansky\The Great Divergence\Results\drg_cpt_ratios_data.csv"


# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_payment_file(filepath, payment_type):
    """Parse payment statistics files and extract mean payments by year."""
    payments = defaultdict(lambda: defaultdict(float))

    with open(filepath, 'r') as f:
        content = f.read()

    if payment_type == 'CPT':
        table_marker = "SUMMARY TABLE - NON-FACILITY MEAN PAYMENTS BY YEAR"
    else:
        table_marker = "SUMMARY TABLE - MEAN PAYMENTS BY DRG AND YEAR"

    table_start = content.find(table_marker)
    if table_start == -1:
        print(f"Warning: Could not find summary table in {filepath}")
        return payments

    data_started = False
    for line in content[table_start:].split('\n'):
        if '----' in line and not data_started:
            data_started = True
            continue
        if not data_started or not line.strip() or line.startswith('='):
            continue

        if payment_type == 'CPT':
            parts = line.split()
            if parts and parts[0].isdigit() and len(parts[0]) == 5:
                code = parts[0]
                values = []
                for part in parts[1:]:
                    if part == '$':
                        continue
                    try:
                        values.append(float(part.replace(',', '')))
                    except ValueError:
                        continue
                if len(values) == 5:
                    for i, yr in enumerate(['2019', '2020', '2021', '2022', '2023']):
                        payments[code][yr] = values[i]

        else:  # DRG
            parts = line.split()
            if parts and parts[0].isdigit():
                code = parts[0]
                dollar_values = []
                i = 0
                while i < len(parts):
                    if parts[i] == '$' and i + 1 < len(parts):
                        try:
                            dollar_values.append(float(parts[i + 1].replace(',', '')))
                            i += 2
                        except ValueError:
                            i += 1
                    else:
                        i += 1
                if len(dollar_values) >= 5:
                    for i, yr in enumerate(['2019', '2020', '2021', '2022', '2023']):
                        if i < len(dollar_values):
                            payments[code][yr] = dollar_values[i]
                elif 'N/A' in line:
                    yr_idx = 0
                    for i, part in enumerate(parts):
                        if part == '$' and i + 1 < len(parts):
                            try:
                                value = float(parts[i + 1].replace(',', ''))
                                yrs = ['2019', '2020', '2021', '2022', '2023']
                                if yr_idx < len(yrs):
                                    payments[code][yrs[yr_idx]] = value
                            except ValueError:
                                pass
                            yr_idx += 1
                        elif part == 'N/A':
                            yr_idx += 1

    return payments


def calculate_ratios(crosswalk_df, cpt_payments, drg_payments):
    """Calculate DRG/CPT ratios based on crosswalk mappings."""
    results = []
    for _, row in crosswalk_df.iterrows():
        if pd.isna(row['CPT']) or pd.isna(row['DRG-Max']) or pd.isna(row['DRG-Min']):
            continue
        cpt      = str(int(row['CPT']))
        drg_max  = str(int(row['DRG-Max']))
        drg_min  = str(int(row['DRG-Min']))
        procedure = row['Procedure']
        proc_type = row['Type']

        if cpt not in cpt_payments:
            print(f"  Note: CPT {cpt} ({procedure}) not found in payment data — skipping")
            continue

        for year in ['2019', '2020', '2021', '2022', '2023']:
            cpt_payment = cpt_payments[cpt].get(year, 0)
            if cpt_payment > 0:
                drg_max_payment = drg_payments[drg_max].get(year, 0)
                drg_min_payment = drg_payments[drg_min].get(year, 0)
                results.append({
                    'Type':            proc_type,
                    'CPT':             cpt,
                    'Procedure':       procedure,
                    'DRG_Max':         drg_max,
                    'DRG_Min':         drg_min,
                    'Year':            int(year),
                    'CPT_Payment':     cpt_payment,
                    'DRG_Max_Payment': drg_max_payment,
                    'DRG_Min_Payment': drg_min_payment,
                    'Ratio_Max':       drg_max_payment / cpt_payment if drg_max_payment > 0 else None,
                    'Ratio_Min':       drg_min_payment / cpt_payment if drg_min_payment > 0 else None,
                })
    return pd.DataFrame(results)


def write_text_results(results_df, output_path):
    """Write detailed ratio results and trend summary to text file."""
    with open(output_path, 'w') as f:
        f.write("DRG/CPT Payment Ratio Analysis (Inflation-Adjusted to 2023 Dollars)\n")
        f.write("=" * 80 + "\n")
        f.write("All payment amounts adjusted to 2023 dollars using CPI-U values.\n")
        f.write("Note: Statistical trend analysis (OLS, Mann-Kendall, BH correction)\n")
        f.write("      is performed in Step 5. Figures are generated in Step 5.\n")
        f.write("=" * 80 + "\n\n")

        f.write("SUMMARY STATISTICS BY YEAR\n")
        f.write("-" * 60 + "\n")
        for year in sorted(results_df['Year'].unique()):
            year_data = results_df[results_df['Year'] == year]
            f.write(f"\nYear {year}:\n")
            f.write(f"  Total procedure mappings analyzed: {len(year_data)}\n")
            max_ratios = year_data['Ratio_Max'].dropna()
            if len(max_ratios) > 0:
                f.write(f"  DRG-Max/CPT Ratios:\n")
                f.write(f"    Mean:   {max_ratios.mean():.1f}x\n")
                f.write(f"    Median: {max_ratios.median():.1f}x\n")
                f.write(f"    Min:    {max_ratios.min():.1f}x\n")
                f.write(f"    Max:    {max_ratios.max():.1f}x\n")
            min_ratios = year_data['Ratio_Min'].dropna()
            if len(min_ratios) > 0:
                f.write(f"  DRG-Min/CPT Ratios:\n")
                f.write(f"    Mean:   {min_ratios.mean():.1f}x\n")
                f.write(f"    Median: {min_ratios.median():.1f}x\n")
                f.write(f"    Min:    {min_ratios.min():.1f}x\n")
                f.write(f"    Max:    {min_ratios.max():.1f}x\n")

        f.write("\n\nDETAILED RESULTS BY PROCEDURE TYPE\n")
        f.write("=" * 80 + "\n")
        for ptype in sorted(results_df['Type'].dropna().unique()):
            type_data = results_df[results_df['Type'] == ptype]
            f.write(f"\n{ptype} Procedures:\n")
            f.write("-" * 40 + "\n")
            procedures = type_data[['CPT', 'Procedure', 'DRG_Max', 'DRG_Min']].drop_duplicates()
            for _, proc in procedures.iterrows():
                f.write(f"\nCPT {proc['CPT']}: {proc['Procedure']}\n")
                f.write(f"DRG Range: {proc['DRG_Min']} - {proc['DRG_Max']}\n")
                f.write("Year  CPT Payment  DRG-Max Payment  DRG-Min Payment  Max Ratio  Min Ratio\n")
                proc_data = type_data[type_data['CPT'] == proc['CPT']].sort_values('Year')
                for _, row in proc_data.iterrows():
                    f.write(f"{row['Year']}  ${row['CPT_Payment']:>10,.0f}  ")
                    f.write(f"${row['DRG_Max_Payment']:>14,.0f}  " if row['DRG_Max_Payment'] > 0 else f"{'N/A':>15}  ")
                    f.write(f"${row['DRG_Min_Payment']:>14,.0f}  " if row['DRG_Min_Payment'] > 0 else f"{'N/A':>15}  ")
                    f.write(f"{row['Ratio_Max']:>9.1f}x  " if row['Ratio_Max'] else f"{'N/A':>10}  ")
                    f.write(f"{row['Ratio_Min']:>9.1f}x\n" if row['Ratio_Min'] else f"{'N/A':>10}\n")

        f.write("\n\nTREND ANALYSIS (2019-2023)\n")
        f.write("=" * 80 + "\n")
        f.write("(Full statistical analysis with BH correction is in Step 5 output)\n\n")
        yearly_avg = results_df.groupby('Year')[['Ratio_Max', 'Ratio_Min']].mean()
        f.write("Year   Max Ratio   Min Ratio   Max Change   Min Change\n")
        f.write("-" * 55 + "\n")
        sorted_years = sorted(yearly_avg.index)
        for i, year in enumerate(sorted_years):
            line = f"{year}   {yearly_avg.loc[year, 'Ratio_Max']:>9.1f}   {yearly_avg.loc[year, 'Ratio_Min']:>9.1f}"
            if i > 0:
                prev = sorted_years[i - 1]
                mc = ((yearly_avg.loc[year, 'Ratio_Max'] / yearly_avg.loc[prev, 'Ratio_Max']) - 1) * 100
                nc = ((yearly_avg.loc[year, 'Ratio_Min'] / yearly_avg.loc[prev, 'Ratio_Min']) - 1) * 100
                line += f"   {mc:>+9.1f}%   {nc:>+9.1f}%"
            f.write(line + "\n")
        max_overall = ((yearly_avg.loc[2023, 'Ratio_Max'] / yearly_avg.loc[2019, 'Ratio_Max']) - 1) * 100
        min_overall = ((yearly_avg.loc[2023, 'Ratio_Min'] / yearly_avg.loc[2019, 'Ratio_Min']) - 1) * 100
        f.write(f"\nOverall Change (2019-2023):\n")
        f.write(f"  Max DRG Ratio: {max_overall:+.1f}%\n")
        f.write(f"  Min DRG Ratio: {min_overall:+.1f}%\n")


def main():
    print("Step 4: DRG/CPT Payment Ratio Calculation (Inflation-Adjusted to 2023 Dollars)")
    print("=" * 70)
    print("NOTE: Figure generation has moved to Step 5.")
    print("=" * 70)

    # Load crosswalk
    print("\nLoading crosswalk data...")
    crosswalk_df = pd.read_csv(crosswalk_file)
    print(f"Loaded {len(crosswalk_df)} crosswalk entries")
    print(crosswalk_df.head())

    # Parse payment files
    print("\nParsing CPT payments...")
    cpt_payments = parse_payment_file(cpt_file, 'CPT')
    print(f"Found payments for {len(cpt_payments)} CPT codes")

    print("\nParsing DRG payments...")
    drg_payments = parse_payment_file(drg_file, 'DRG')
    print(f"Found payments for {len(drg_payments)} DRG codes")

    # Calculate ratios
    print("\nCalculating DRG/CPT ratios...")
    results_df = calculate_ratios(crosswalk_df, cpt_payments, drg_payments)
    print(f"Calculated {len(results_df)} ratio entries")

    if results_df.empty:
        print("\nNo matching CPT-DRG pairs found. Checking crosswalk vs payment data...")
        crosswalk_cpts = crosswalk_df['CPT'].dropna().astype(int).astype(str).tolist()
        missing = [c for c in crosswalk_cpts if c not in cpt_payments]
        print(f"Missing CPTs: {missing}")
        return

    # Save ratio data CSV for Step 5
    results_df.to_csv(ratios_csv, index=False)
    print(f"\nRatio data saved for Step 5: {ratios_csv}")

    # Write detailed text results
    print("\nWriting detailed text results...")
    write_text_results(results_df, output_file)
    print(f"Text results saved to: {output_file}")

    print("\nStep 4 complete. Run Step 5 to generate statistical analysis and all figures.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    input("\nPress Enter to close...")
