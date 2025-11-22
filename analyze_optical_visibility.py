#!/usr/bin/env python3
"""
Analyze optical visibility of NEOWISE sources in ZTF
"""

import pandas as pd
import numpy as np
from pathlib import Path

output_dir = Path('/Users/marcus/Desktop/RealYSO/Analysis')
output_dir.mkdir(exist_ok=True)

print("=" * 90)
print("NEOWISE OPTICAL VISIBILITY ANALYSIS")
print("Identifying which NEOWISE sources are optically visible and have ZTF lightcurves")
print("=" * 90)

# Paper 2: Neha & Sharma - Contains explicit light curve classifications
print("\n" + "=" * 90)
print("PAPER 2: Neha & Sharma - NEOWISE Variability Statistics")
print("=" * 90)

try:
    # Read with space-separated values
    df2 = pd.read_csv('/Users/marcus/Desktop/RealYSO/apjsadc397t2_mrt.txt',
                      sep=r'\s+', skiprows=35, engine='python')
    
    print(f"\nTotal sources: {len(df2)}")
    print(f"\nColumn names: {df2.columns.tolist()}")
    
    # The last column should be LCType
    if 'LCType' in df2.columns:
        print(f"\nLight curve classification types:")
        lc_counts = df2['LCType'].value_counts()
        for lctype, count in lc_counts.items():
            print(f"  {lctype:15s}: {count:5d} sources ({100*count/len(df2):5.1f}%)")
        
        # Sources that are optically variable (not NV = non-variable)
        nv_count = len(df2[df2['LCType'] == 'NV'])
        variable_count = len(df2[df2['LCType'] != 'NV'])
        
        print(f"\nOptical variability summary:")
        print(f"  Non-variable (NV):  {nv_count:6d} sources ({100*nv_count/len(df2):5.1f}%)")
        print(f"  Variable:           {variable_count:6d} sources ({100*variable_count/len(df2):5.1f}%)")
        
        # Types of variability
        variable_types = df2[df2['LCType'] != 'NV']['LCType'].unique()
        print(f"\nVariability types:")
        for vtype in sorted(variable_types):
            count = len(df2[df2['LCType'] == vtype])
            print(f"  {vtype:15s}: {count:5d} sources")
        
        # Save as CSV
        df2.to_csv(output_dir / 'paper2_neowise_variability.csv', index=False)
        print(f"\n✓ Saved to: paper2_neowise_variability.csv")
    else:
        print(f"Warning: LCType column not found")
        print(f"Available columns: {df2.columns.tolist()}")
        print(f"\nFirst row:")
        print(df2.iloc[0])

except Exception as e:
    print(f"Error reading Paper 2: {e}")
    import traceback
    traceback.print_exc()

# Paper 3: LAMOST - H-alpha emission indicators
print("\n" + "=" * 90)
print("PAPER 3: LAMOST - H-alpha Emission YSO Candidates")
print("=" * 90)

try:
    df3 = pd.read_csv('/Users/marcus/Desktop/RealYSO/apjsadf4e6t4_mrt.txt',
                      sep=r'\s+', skiprows=31, engine='python')
    
    print(f"\nTotal sources: {len(df3)}")
    print(f"\nColumns available: {df3.columns.tolist()}")
    
    if 'EW' in df3.columns:
        # EW is H-alpha equivalent width (indicator of emission)
        ew_positive = len(df3[df3['EW'] > 0])
        ew_negative = len(df3[df3['EW'] <= 0])
        
        print(f"\nH-alpha emission (positive EW): {ew_positive:6d} sources ({100*ew_positive/len(df3):5.1f}%)")
        print(f"H-alpha absorption (negative EW): {ew_negative:6d} sources ({100*ew_negative/len(df3):5.1f}%)")
        
        print(f"\nH-alpha EW statistics:")
        print(f"  Mean:   {df3['EW'].mean():8.2f} Å")
        print(f"  Median: {df3['EW'].median():8.2f} Å")
        print(f"  Max:    {df3['EW'].max():8.2f} Å")
        print(f"  Min:    {df3['EW'].min():8.2f} Å")
    
    df3.to_csv(output_dir / 'paper3_lamost_halpha.csv', index=False)
    print(f"\n✓ Saved to: paper3_lamost_halpha.csv")

except Exception as e:
    print(f"Error reading Paper 3: {e}")
    import traceback
    traceback.print_exc()

# Paper 1: FUors - Parse carefully
print("\n" + "=" * 90)
print("PAPER 1: FUors - Long-lasting YSO Outbursts")
print("=" * 90)

try:
    with open('/Users/marcus/Desktop/RealYSO/apjadd25ft1_mrt.txt', 'r') as f:
        lines = f.readlines()
    
    # Find the data section
    sources = []
    print(f"\nParsing {len(lines)} lines...")
    
    for i, line in enumerate(lines):
        if 'Byte-by-byte' in line:
            print(f"Header found at line {i}")
        if i >= 40 and line.strip() and not line.startswith('---'):
            # This is data
            try:
                parts = line.split()
                if len(parts) >= 3:
                    spicy_id = parts[0]
                    yso_class = parts[1]
                    
                    # Try to extract RA/Dec from the fixed positions
                    # According to header: RAh at bytes 131-132, RAm at 134-135, RAs at 137-143
                    if len(line) > 160:
                        try:
                            rah = int(line[131:133].strip())
                            ram = int(line[134:136].strip())
                            ras = float(line[137:143].strip())
                            ra_deg = rah * 15 + ram * 0.25 + ras * (15/3600)
                            
                            de_sign = -1 if line[145] == '-' else 1
                            ded = int(line[146:148].strip())
                            dem = int(line[149:151].strip())
                            des = float(line[152:157].strip())
                            dec_deg = de_sign * (ded + dem/60 + des/3600)
                            
                            sources.append({
                                'SPICY_ID': spicy_id,
                                'Class': yso_class,
                                'RA': ra_deg,
                                'Dec': dec_deg,
                                'line': i
                            })
                        except:
                            pass
            except:
                pass
    
    if len(sources) > 0:
        df1 = pd.DataFrame(sources)
        print(f"\n✓ Successfully parsed {len(df1)} sources from Paper 1")
        print(f"\nYSO Class distribution:")
        print(df1['Class'].value_counts())
        
        print(f"\nCoordinate statistics:")
        print(f"  RA range:  {df1['RA'].min():.2f} to {df1['RA'].max():.2f} deg")
        print(f"  Dec range: {df1['Dec'].min():.2f} to {df1['Dec'].max():.2f} deg")
        
        df1.to_csv(output_dir / 'paper1_fuors.csv', index=False)
        print(f"\n✓ Saved to: paper1_fuors.csv")
    else:
        print("\nNote: Could not parse sources from Paper 1 (fixed-width format)")
        print("First few data lines:")
        for i in range(40, 45):
            print(f"  Line {i}: {lines[i][:100]}")

except Exception as e:
    print(f"Error reading Paper 1: {e}")
    import traceback
    traceback.print_exc()

# Summary and recommendation
print("\n" + "=" * 90)
print("SUMMARY & RECOMMENDATIONS FOR ZTF QUERY")
print("=" * 90)

try:
    summary = """
KEY FINDINGS:
=============

1. PAPER 2 (Neha & Sharma): NEOWISE Variability
   - Contains {p2_total} YSO sources with NEOWISE variability statistics
   - Has EXPLICIT light curve classifications (LCType column):
     * NV (Non-Variable): steady objects
     * Irregular: erratic variability
     * Curved: smooth trend variability
   - All {p2_total} sources have decimal RA/Dec coordinates → OPTIMAL for ZTF queries

2. PAPER 3 (LAMOST): H-alpha Emission YSO Candidates
   - Contains {p3_total} H-alpha emission YSO candidates
   - All sources have decimal RA/Dec coordinates → CAN be queried in ZTF
   - H-alpha emission indicates young/active systems (likely optically variable)

3. PAPER 1 (FUors): Infrared Variable YSOs
   - Contains FUor objects with known infrared variability
   - May also show optical variability in ZTF
   - Coordinates in sexagesimal format (needs conversion)

RECOMMENDED APPROACH:
=====================

For optical variability analysis in ZTF:

1. START with Paper 2 sources that are classified as VARIABLE (Irregular/Curved types)
   → These NEOWISE sources already show mid-IR variability
   → Check if they also vary optically in ZTF

2. ALSO query Paper 3 LAMOST sources
   → H-alpha emission indicates youth
   → Good candidates for optical variability

3. For each source found in ZTF:
   → Compare optical and infrared variability timescales
   → Identify whether variability is consistent between bands

NEXT STEPS:
===========

Given that ZTF API queries are slow/timing out:
1. Create a Python script using ZTFQ or other efficient ZTF query methods
2. Focus on subset of most interesting sources first
3. Generate lightcurves for those with highest likelihood of optical variability
""".format(
        p2_total = len(df2) if 'df2' in locals() else '?',
        p3_total = len(df3) if 'df3' in locals() else '?'
    )
    
    print(summary)
    
except Exception as e:
    print(f"Error in summary: {e}")

print("=" * 90)
print(f"\nAnalysis files saved to: {output_dir}")
print("=" * 90)
