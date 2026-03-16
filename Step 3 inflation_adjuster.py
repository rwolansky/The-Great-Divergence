import re
import os

# CPI-U values from R-CPI-U data
cpi_values = {
    2019: 375.8,
    2020: 380.8,
    2021: 399.2,
    2022: 431.5,
    2023: 449.3
}

# Base year for inflation adjustment (2023)
base_year = 2023
base_cpi = cpi_values[base_year]

def calculate_inflation_factor(year):
    """Calculate inflation adjustment factor to convert to 2023 dollars"""
    if year not in cpi_values:
        raise ValueError(f"CPI data not available for year {year}")
    return base_cpi / cpi_values[year]

def adjust_dollar_amount(amount_str, year):
    """Adjust a dollar amount string for inflation"""
    # Remove $ and commas, convert to float
    amount = float(amount_str.replace('$', '').replace(',', '').strip())
    # Adjust for inflation
    adjusted = amount * calculate_inflation_factor(year)
    # Return formatted string
    return f"${adjusted:,.2f}"

def process_cpt_file(input_path, output_path):
    """Process CPT payment statistics file and adjust for inflation"""
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    output_lines = []
    current_year = None
    
    # Add header about inflation adjustment
    output_lines.append("Medicare Physician Fee Schedule Payment Statistics (Adjusted to 2023 Dollars)\n")
    output_lines.append("=" * 80 + "\n")
    output_lines.append("Note: All dollar amounts have been adjusted for inflation using CPI-U values\n")
    output_lines.append("Original values were in nominal dollars for each respective year\n")
    
    for line in lines[4:]:  # Skip original header
        # Check if this is a year header
        year_match = re.match(r'^YEAR (\d{4})', line)
        if year_match:
            current_year = int(year_match.group(1))
            factor = calculate_inflation_factor(current_year)
            output_lines.append(f"\nYEAR {current_year} (Inflation factor: {factor:.3f})\n")
        # Check if this line contains dollar amounts
        elif current_year and '$' in line:
            # Find all dollar amounts in the line
            dollar_pattern = r'\$[\d,]+\.?\d*'
            
            def replace_amount(match):
                return adjust_dollar_amount(match.group(), current_year)
            
            adjusted_line = re.sub(dollar_pattern, replace_amount, line)
            output_lines.append(adjusted_line)
        else:
            output_lines.append(line)
    
    with open(output_path, 'w') as f:
        f.writelines(output_lines)
    
    print(f"CPT file processed. Output saved to: {output_path}")

def process_drg_file(input_path, output_path):
    """Process DRG payment statistics file and adjust for inflation"""
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    output_lines = []
    current_year = None
    
    # Add header about inflation adjustment
    output_lines.append("Medicare DRG Payment Statistics Analysis (Adjusted to 2023 Dollars)\n")
    output_lines.append("=" * 80 + "\n")
    output_lines.append("Note: All dollar amounts have been adjusted for inflation using CPI-U values\n")
    output_lines.append("Original values were in nominal dollars for each respective year\n")
    
    for line in lines[4:]:  # Skip original header
        # Check if this is a year header
        if line.strip().startswith('YEAR '):
            year_match = re.search(r'YEAR (\d{4})', line)
            if year_match:
                current_year = int(year_match.group(1))
                factor = calculate_inflation_factor(current_year)
                output_lines.append(f"\n{'='*60}\n")
                output_lines.append(f"YEAR {current_year} (Inflation factor: {factor:.3f})\n")
                output_lines.append(f"{'='*60}\n")
        # Check if this line contains dollar amounts
        elif current_year and '$' in line:
            # Find all dollar amounts in the line
            dollar_pattern = r'\$[\d,]+\.?\d*'
            
            def replace_amount(match):
                return adjust_dollar_amount(match.group(), current_year)
            
            adjusted_line = re.sub(dollar_pattern, replace_amount, line)
            output_lines.append(adjusted_line)
        else:
            output_lines.append(line)
    
    with open(output_path, 'w') as f:
        f.writelines(output_lines)
    
    print(f"DRG file processed. Output saved to: {output_path}")

def main():
    """Main function to process both files"""
    # Define file paths
    cpt_input = r"Z:\homes\Rachel Wolansky\The Great Divergence\Results\payment_statistics.txt"
    cpt_output = r"Z:\homes\Rachel Wolansky\The Great Divergence\Results\payment_statistics_inflation_adjusted.txt"
    
    drg_input = r"Z:\homes\Rachel Wolansky\The Great Divergence\Results\drg_payment_statistics.txt"
    drg_output = r"Z:\homes\Rachel Wolansky\The Great Divergence\Results\drg_payment_statistics_inflation_adjusted.txt"
    
    # Display inflation factors
    print("Inflation Adjustment Factors (to 2023 dollars):")
    print("-" * 40)
    for year in sorted(cpi_values.keys()):
        factor = calculate_inflation_factor(year)
        print(f"{year}: {factor:.3f} ({(factor-1)*100:+.1f}%)")
    print()
    
    # Process CPT file
    if os.path.exists(cpt_input):
        print("Processing CPT payment statistics file...")
        process_cpt_file(cpt_input, cpt_output)
    else:
        print(f"CPT file not found: {cpt_input}")
    
    # Process DRG file
    if os.path.exists(drg_input):
        print("Processing DRG payment statistics file...")
        process_drg_file(drg_input, drg_output)
    else:
        print(f"DRG file not found: {drg_input}")
    
    print("\nProcessing complete!")
    print("\nInflation-adjusted files created:")
    print(f"- {cpt_output}")
    print(f"- {drg_output}")

if __name__ == "__main__":
    main()
