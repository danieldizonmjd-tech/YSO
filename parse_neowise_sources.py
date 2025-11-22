#!/usr/bin/env python3
"""
Parse NEOWISE data files and identify optically visible sources
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

output_dir = Path('/Users/marcus/Desktop/RealYSO/Analysis')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("NEOWISE Source Catalog Analysis")
print("=" * 80)

# Paper 2: Neha & Sharma - Variability statistics
print("\n" + "=" * 80)
print("Paper 2: Neha & Sharma (apjsadc397t2_mrt.txt)")
print("Variability statistics of individual objects")
print("=" * 80)

try:
    df2 = pd.read_csv('/Users/marcus/Desktop/RealYSO/apjsadc397t2_mrt.txt',
                      delim_whitespace=True, skiprows=35)
    print(f"\nFound {len(df2)} sources")
    print(f"Columns: {df2.columns.tolist()}")
    print(f"\nData types:")
    print(df2.dtypes)
    print(f"\nFirst 5 sources:")
    print(df2[['Objname', 'RAdeg', 'DEdeg', 'YSO-CLASS', 'LCType']].head(10))
    
    # Check for optical variability indicators
    print(f"\nLight curve types in Paper 2:")
    print(df2['LCType'].value_counts())
    
    df2.to_csv(output_dir / 'paper2_sources.csv', index=False)
    print(f"\n✓ Saved {len(df2)} sources to paper2_sources.csv")
    
except Exception as e:
    print(f"Error parsing Paper 2: {e}")
    import traceback
    traceback.print_exc()

# Paper 3: LAMOST YSOs
print("\n" + "=" * 80)
print("Paper 3: LAMOST (apjsadf4e6t4_mrt.txt)")
print("H-alpha Emission Line Stars and YSO Candidates")
print("=" * 80)

try:
    df3 = pd.read_csv('/Users/marcus/Desktop/RealYSO/apjsadf4e6t4_mrt.txt',
                      delim_whitespace=True, skiprows=31)
    print(f"\nFound {len(df3)} sources")
    print(f"Columns: {df3.columns.tolist()}")
    print(f"\nFirst 5 sources:")
    if 'Design' in df3.columns:
        print(df3[['Design', 'RAdeg', 'DEdeg', 'EW']].head(10))
    else:
        print(df3.head())
    
    df3.to_csv(output_dir / 'paper3_sources.csv', index=False)
    print(f"\n✓ Saved {len(df3)} sources to paper3_sources.csv")
    
except Exception as e:
    print(f"Error parsing Paper 3: {e}")
    import traceback
    traceback.print_exc()

# Paper 1: FUors - Fixed width format
print("\n" + "=" * 80)
print("Paper 1: FUors (apjadd25ft1_mrt.txt)")
print("Long-lasting YSO outbursts (infrared variability)")
print("=" * 80)

try:
    with open('/Users/marcus/Desktop/RealYSO/apjadd25ft1_mrt.txt', 'r') as f:
        lines = f.readlines()
    
    # Read header to understand format
    print("\nHeader (lines 7-39):")
    for i in range(7, 39):
        print(lines[i].rstrip())
    
    # Parse data starting from line 40
    sources = []
    print("\nParsing data...")
    
    for line_num, line in enumerate(lines[40:], start=41):
        if len(line.strip()) == 0:
            continue
        
        try:
            # Parse according to byte positions from header
            # RAh RAm RAs: bytes 131-143 (space-padded)
            # DE- DEd DEm DEs: bytes 145-157
            
            # Extract fields
            if len(line) > 160:
                spicy_id = line[0:6].strip()
                yso_class = line[8:14].strip()
                i1mag = float(line[15:21].strip()) if line[15:21].strip() else None
                w1mag = float(line[41:47].strip()) if line[41:47].strip() else None
                w2mag = float(line[54:60].strip()) if line[54:60].strip() else None
                
                # Sexagesimal coordinates
                rah = int(line[131:133].strip()) if line[131:133].strip() else 0
                ram = int(line[134:136].strip()) if line[134:136].strip() else 0
                ras = float(line[137:143].strip()) if line[137:143].strip() else 0
                
                de_sign = -1 if line[145] == '-' else 1
                ded = int(line[146:148].strip()) if line[146:148].strip() else 0
                dem = int(line[149:151].strip()) if line[149:151].strip() else 0
                des = float(line[152:157].strip()) if line[152:157].strip() else 0
                
                # Convert to decimal degrees
                ra_deg = rah * 15 + ram * 0.25 + ras * (15/3600)
                dec_deg = de_sign * (ded + dem/60 + des/3600)
                
                sources.append({
                    'SPICY_ID': spicy_id,
                    'Class': yso_class,
                    'I1_mag': i1mag,
                    'W1_mag': w1mag,
                    'W2_mag': w2mag,
                    'RA_deg': ra_deg,
                    'Dec_deg': dec_deg
                })
        
        except Exception as e:
            pass  # Skip malformed lines
    
    if len(sources) > 0:
        df1 = pd.DataFrame(sources)
        print(f"\nFound {len(df1)} sources")
        print(f"\nFirst 5 sources:")
        print(df1.head())
        print(f"\nYSO Classes:")
        print(df1['Class'].value_counts())
        
        df1.to_csv(output_dir / 'paper1_sources.csv', index=False)
        print(f"\n✓ Saved {len(df1)} sources to paper1_sources.csv")
    else:
        print("\nNo sources parsed from Paper 1")

except Exception as e:
    print(f"Error parsing Paper 1: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

try:
    p1_sources = len(df1) if 'df1' in locals() else 0
    p2_sources = len(df2) if 'df2' in locals() else 0
    p3_sources = len(df3) if 'df3' in locals() else 0
    total = p1_sources + p2_sources + p3_sources
    
    print(f"\nTotal sources across all catalogs: {total}")
    print(f"  Paper 1 (FUors): {p1_sources} sources with infrared variability")
    print(f"  Paper 2 (Neha & Sharma): {p2_sources} sources with NEOWISE variability")
    print(f"  Paper 3 (LAMOST): {p3_sources} H-alpha emission YSO candidates")
    
    print(f"\nOptical visibility:")
    print(f"  Paper 2 light curve types:")
    if 'df2' in locals():
        for lctype in df2['LCType'].value_counts().items():
            print(f"    {lctype[0]}: {lctype[1]} sources")
    
    print(f"\nOutput files saved to: {output_dir}")
    print(f"  - paper1_sources.csv")
    print(f"  - paper2_sources.csv")
    print(f"  - paper3_sources.csv")
    
except Exception as e:
    print(f"Error in summary: {e}")

print("\n" + "=" * 80)
