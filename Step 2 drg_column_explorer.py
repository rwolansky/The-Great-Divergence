import os
os.chdir(r"Z:\homes\Rachel Wolansky\The Great Divergence\Results")
import pandas as pd
import numpy as np
from collections import defaultdict

# Define the DRG codes to analyze (removing duplicates)
target_drgs = sorted(list(set([
    '339', '343', '417', '419', '329', '331',
    '388', '390', '619', '621', '405', '407', '650', '652', 
    '005', '006', '001', '002', '268', '269'
])))

# Convert to both regular and zero-padded versions since DRGs might be stored either way
target_drgs_all = []
for drg in target_drgs:
    target_drgs_all.append(drg.lstrip('0'))  # Remove leading zeros
    target_drgs_all.append(drg.zfill(3))     # Pad with zeros to 3 digits
target_drgs_all = list(set(target_drgs_all))

# Define file paths
base_path = r"Z:\homes\Rachel Wolansky\The Great Divergence\Data In\Medicare Inpatient Hospital Data"
years = ['2019', '2020', '2021', '2022', '2023']
output_file = r"Z:\homes\Rachel Wolansky\The Great Divergence\Results\drg_payment_statistics.txt"

def calculate_statistics(values):
    """Calculate min, max, mean, median, and std dev for a list of values"""
    if not values or len(values) == 0:
        return {
            'min': 'N/A',
            'max': 'N/A',
            'mean': 'N/A',
            'median': 'N/A',
            'std_dev': 'N/A',
            'count': 0
        }
    
    arr = np.array(values)
    return {
        'min': f"${np.min(arr):,.2f}",
        'max': f"${np.max(arr):,.2f}",
        'mean': f"${np.mean(arr):,.2f}",
        'median': f"${np.median(arr):,.2f}",
        'std_dev': f"${np.std(arr):,.2f}",
        'count': len(values)
    }

# Main processing
results = defaultdict(lambda: defaultdict(list))
drg_descriptions = {}

print("Processing DRG payment files...")
print(f"Looking for DRGs: {', '.join(sorted(set([d.lstrip('0') for d in target_drgs])))}")

for year in years:
    filename = f"medicare_inpatient_hospital_data_{year}.csv"
    filepath = os.path.join(base_path, filename)
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found - {filepath}")
        continue
    
    print(f"\nProcessing {filename}...")
    
    try:
        df = pd.read_csv(filepath, dtype={'DRG_Cd': str})
        
        drg_found_in_year = set()
        
        for idx, row in df.iterrows():
            try:
                drg_code = str(row['DRG_Cd']).strip()
                
                if drg_code in target_drgs_all:
                    drg_normalized = drg_code.lstrip('0')
                    if not drg_normalized:
                        drg_normalized = '0'
                    
                    if drg_normalized not in drg_descriptions and pd.notna(row['DRG_Desc']):
                        drg_descriptions[drg_normalized] = str(row['DRG_Desc'])
                    
                    payment = row['Avg_Tot_Pymt_Amt']
                    if pd.notna(payment) and payment > 0:
                        results[year][drg_normalized].append(float(payment))
                        drg_found_in_year.add(drg_normalized)
                        
            except Exception as e:
                continue
        
        print(f"  Found data for {len(drg_found_in_year)} DRGs: {', '.join(sorted(drg_found_in_year))}")
        
    except Exception as e:
        print(f"ERROR reading file: {str(e)}")

# Write results to file
print(f"\nWriting results to {output_file}...")

with open(output_file, 'w') as f:
    f.write("Medicare DRG Payment Statistics Analysis\n")
    f.write("=" * 80 + "\n")
    f.write(f"DRG Codes Analyzed: {', '.join(sorted(set([d.lstrip('0') for d in target_drgs])))}\n")
    f.write(f"Years: 2019-2023\n")
    f.write(f"Data Source: Medicare Inpatient Hospital Data\n")
    f.write("=" * 80 + "\n\n")
    
    display_drgs = sorted(set([d.lstrip('0') if d.lstrip('0') else '0' for d in target_drgs]))
    
    for year in years:
        f.write(f"\n{'='*60}\n")
        f.write(f"YEAR {year}\n")
        f.write(f"{'='*60}\n")
        
        drgs_with_data = []
        drgs_without_data = []
        
        for drg in display_drgs:
            if drg in results[year] and results[year][drg]:
                drgs_with_data.append(drg)
            else:
                drgs_without_data.append(drg)
        
        for drg in sorted(drgs_with_data):
            stats = calculate_statistics(results[year][drg])
            desc = drg_descriptions.get(drg, "No description available")
            
            f.write(f"\nDRG {drg}: {desc[:60]}{'...' if len(desc) > 60 else ''}\n")
            f.write(f"  Number of hospitals: {stats['count']}\n")
            f.write(f"  Min payment:         {stats['min']}\n")
            f.write(f"  Max payment:         {stats['max']}\n")
            f.write(f"  Mean payment:        {stats['mean']}\n")
            f.write(f"  Median payment:      {stats['median']}\n")
            f.write(f"  Std deviation:       {stats['std_dev']}\n")
        
        if drgs_without_data:
            f.write(f"\nDRGs with no data in {year}: {', '.join(sorted(drgs_without_data))}\n")
    
    f.write("\n\n" + "=" * 100 + "\n")
    f.write("SUMMARY TABLE - MEAN PAYMENTS BY DRG AND YEAR\n")
    f.write("=" * 100 + "\n")
    f.write(f"{'DRG':<6} {'Description':<40} {'2019':<12} {'2020':<12} {'2021':<12} {'2022':<12} {'2023':<12}\n")
    f.write("-" * 100 + "\n")
    
    for drg in display_drgs:
        desc = drg_descriptions.get(drg, "N/A")[:38]
        row = f"{drg:<6} {desc:<40}"
        
        for year in years:
            if drg in results[year] and results[year][drg]:
                mean_val = np.mean(results[year][drg])
                row += f" ${mean_val:>10,.0f}"
            else:
                row += f" {'N/A':>11}"
        f.write(row + "\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("OVERALL STATISTICS\n")
    f.write("=" * 80 + "\n")
    
    for year in years:
        total_drgs = len([d for d in display_drgs if d in results[year] and results[year][d]])
        total_payments = sum(len(results[year][d]) for d in results[year])
        
        f.write(f"\n{year}:")
        f.write(f"\n  DRGs with data: {total_drgs} out of {len(display_drgs)}")
        f.write(f"\n  Total hospital-DRG combinations: {total_payments}")

print("\nProcessing complete!")
print(f"Results saved to: {output_file}")

print("\nSummary of DRGs found across all years:")
all_drgs_found = set()
for year in years:
    all_drgs_found.update(results[year].keys())
print(f"Total unique DRGs found: {len(all_drgs_found)}")
print(f"DRGs found: {', '.join(sorted(all_drgs_found))}")
