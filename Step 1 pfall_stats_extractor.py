import os
os.chdir(r"Z:\homes\Rachel Wolansky\The Great Divergence\Results")
import numpy as np
from collections import defaultdict

# Define the CPT codes to analyze
target_cpts = [
    '44970', '47562', '44140', '44204', '44206', '44143',
    '44120', '44005', '44180', '43644', '43775', '47120', '50360', '47135',
    '33945', '34701'
]

# Define file paths
base_path = r"Z:\homes\Rachel Wolansky\The Great Divergence\Data In\PFALL"
years = ['19', '20', '21', '22', '23']
output_file = r"Z:\homes\Rachel Wolansky\The Great Divergence\Results\payment_statistics.txt"

def parse_pfall_record(line):
    """Parse a single PFALL record based on the fixed-width format"""
    try:
        # Extract fields based on documented positions
        year = line[1:5].strip()
        carrier = line[8:13].strip()
        locality = line[16:18].strip()
        hcpcs = line[21:26].strip()
        modifier = line[29:31].strip()
        
        # Extract payment amounts (positions are 0-based, so subtract 1)
        non_facility_fee = line[34:44].strip()
        facility_fee = line[47:57].strip()
        
        # Convert fees to float, handling empty values
        non_facility = float(non_facility_fee) if non_facility_fee else None
        facility = float(facility_fee) if facility_fee else None
        
        return {
            'year': year,
            'carrier': carrier,
            'locality': locality,
            'hcpcs': hcpcs,
            'modifier': modifier,
            'non_facility_fee': non_facility,
            'facility_fee': facility
        }
    except:
        return None

def calculate_statistics(values):
    """Calculate min, max, mean, median, and std dev for a list of values"""
    if not values:
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
        'min': f"{np.min(arr):.2f}",
        'max': f"{np.max(arr):.2f}",
        'mean': f"{np.mean(arr):.2f}",
        'median': f"{np.median(arr):.2f}",
        'std_dev': f"{np.std(arr):.2f}",
        'count': len(values)
    }

# Main processing
results = defaultdict(lambda: defaultdict(lambda: {'non_facility': [], 'facility': []}))

print("Processing PFALL files...")

for year in years:
    filename = f"PFALL{year}.txt"
    filepath = os.path.join(base_path, filename)
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found - {filepath}")
        continue
    
    print(f"Processing {filename}...")
    
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            # Skip header/trailer records
            if line.startswith('"TRL') or len(line) < 100:
                continue
                
            record = parse_pfall_record(line)
            if record and record['hcpcs'] in target_cpts:
                cpt = record['hcpcs']
                full_year = f"20{year}"
                
                # Only include if no modifier (blank) for global fees
                if record['modifier'] == '':
                    if record['non_facility_fee'] is not None:
                        results[full_year][cpt]['non_facility'].append(record['non_facility_fee'])
                    if record['facility_fee'] is not None:
                        results[full_year][cpt]['facility'].append(record['facility_fee'])

# Write results to file
print(f"\nWriting results to {output_file}...")

with open(output_file, 'w') as f:
    f.write("Medicare Physician Fee Schedule Payment Statistics\n")
    f.write("=" * 80 + "\n")
    f.write(f"CPT Codes Analyzed: {', '.join(target_cpts)}\n")
    f.write(f"Years: 2019-2023\n")
    f.write("=" * 80 + "\n\n")
    
    for year in ['2019', '2020', '2021', '2022', '2023']:
        f.write(f"\nYEAR {year}\n")
        f.write("-" * 60 + "\n")
        
        for cpt in sorted(target_cpts):
            if cpt in results[year]:
                f.write(f"\nCPT {cpt}:\n")
                
                # Non-facility statistics
                nf_stats = calculate_statistics(results[year][cpt]['non_facility'])
                f.write(f"  Non-Facility Fees (n={nf_stats['count']}):\n")
                f.write(f"    Min: ${nf_stats['min']}\n")
                f.write(f"    Max: ${nf_stats['max']}\n")
                f.write(f"    Mean: ${nf_stats['mean']}\n")
                f.write(f"    Median: ${nf_stats['median']}\n")
                f.write(f"    Std Dev: ${nf_stats['std_dev']}\n")
                
                # Facility statistics
                f_stats = calculate_statistics(results[year][cpt]['facility'])
                f.write(f"  Facility Fees (n={f_stats['count']}):\n")
                f.write(f"    Min: ${f_stats['min']}\n")
                f.write(f"    Max: ${f_stats['max']}\n")
                f.write(f"    Mean: ${f_stats['mean']}\n")
                f.write(f"    Median: ${f_stats['median']}\n")
                f.write(f"    Std Dev: ${f_stats['std_dev']}\n")
            else:
                f.write(f"\nCPT {cpt}: No data found\n")
    
    # Summary table
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("SUMMARY TABLE - NON-FACILITY MEAN PAYMENTS BY YEAR\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'CPT':<8} {'2019':<10} {'2020':<10} {'2021':<10} {'2022':<10} {'2023':<10}\n")
    f.write("-" * 58 + "\n")
    
    for cpt in sorted(target_cpts):
        row = f"{cpt:<8}"
        for year in ['2019', '2020', '2021', '2022', '2023']:
            if cpt in results[year] and results[year][cpt]['non_facility']:
                mean_val = np.mean(results[year][cpt]['non_facility'])
                row += f" ${mean_val:>8.2f}"
            else:
                row += f" {'N/A':>9}"
        f.write(row + "\n")

print("Processing complete!")
print(f"Results saved to: {output_file}")

# Display summary of what was found
print("\nSummary of records found:")
for year in ['2019', '2020', '2021', '2022', '2023']:
    total_cpts = sum(1 for cpt in target_cpts if cpt in results[year])
    print(f"  {year}: {total_cpts} CPT codes with data")
