#!/usr/bin/env python3
"""
Extract NEOWISE sources suitable for ZTF analysis and create query list
"""

import pandas as pd
from pathlib import Path

output_dir = Path('/Users/marcus/Desktop/RealYSO/Analysis')
output_dir.mkdir(exist_ok=True)

print("=" * 90)
print("Extracting NEOWISE Sources for ZTF Analysis")
print("=" * 90)

# Paper 2: Read with fixed column widths based on the byte positions in header
print("\n" + "=" * 90)
print("PAPER 2: Neha & Sharma - NEOWISE Variability")
print("=" * 90)

try:
    # Read the header and data separately
    with open('/Users/marcus/Desktop/RealYSO/apjsadc397t2_mrt.txt', 'r') as f:
        lines = f.readlines()
    
    # Find data start
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith('J') and 'deg' not in line:
            data_start = i
            break
    
    print(f"Data starts at line {data_start}")
    
    # Parse each line manually
    sources = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        
        parts = line.split()
        if len(parts) < 5:
            continue
        
        try:
            objname = parts[0]
            ra = float(parts[1])
            dec = float(parts[2])
            sed_slope = float(parts[3])
            yso_class = parts[4]
            
            # Last column is LCType
            lctype = parts[-1] if len(parts) > 20 else 'Unknown'
            
            sources.append({
                'Name': objname,
                'RA': ra,
                'Dec': dec,
                'SED_Slope': sed_slope,
                'YSO_Class': yso_class,
                'LCType': lctype
            })
        except Exception as e:
            pass
    
    df2 = pd.DataFrame(sources)
    
    if len(df2) > 0:
        print(f"\n✓ Successfully extracted {len(df2)} sources")
        
        print(f"\nLight Curve Type Distribution:")
        lc_dist = df2['LCType'].value_counts()
        for lctype, count in lc_dist.items():
            pct = 100 * count / len(df2)
            print(f"  {lctype:15s}: {count:6d} ({pct:5.1f}%)")
        
        print(f"\nYSO Class Distribution:")
        class_dist = df2['YSO_Class'].value_counts()
        for yclass, count in class_dist.items():
            pct = 100 * count / len(df2)
            print(f"  {yclass:10s}: {count:6d} ({pct:5.1f}%)")
        
        # Filter for optically variable sources
        variable = df2[df2['LCType'] != 'NV']
        print(f"\nVariable sources (LCType != NV): {len(variable)}")
        
        # Save full catalog
        df2.to_csv(output_dir / 'paper2_all_sources.csv', index=False)
        
        # Save variable sources
        variable.to_csv(output_dir / 'paper2_variable_sources.csv', index=False)
        
        print(f"\n✓ Saved to:")
        print(f"  - paper2_all_sources.csv ({len(df2)} sources)")
        print(f"  - paper2_variable_sources.csv ({len(variable)} sources)")
        
        # Detailed look at first few sources
        print(f"\nFirst 10 sources:")
        print(df2.head(10).to_string())

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Paper 3: LAMOST H-alpha sources
print("\n" + "=" * 90)
print("PAPER 3: LAMOST - H-alpha Emission YSO Candidates")
print("=" * 90)

try:
    with open('/Users/marcus/Desktop/RealYSO/apjsadf4e6t4_mrt.txt', 'r') as f:
        lines = f.readlines()
    
    # Find data start
    data_start = None
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith('--') and not line.startswith('Byte') \
           and not line.startswith('Note') and not line.startswith('   ') \
           and not line.startswith('Title') and not line.startswith('Authors') \
           and not line.startswith('Table'):
            # Check if looks like LAMOST data
            parts = line.split()
            if len(parts) > 5 and parts[1].startswith('J'):
                data_start = i
                break
    
    if data_start:
        print(f"Data starts at line {data_start}")
        
        sources = []
        for line in lines[data_start:]:
            if not line.strip():
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            try:
                obsid = int(parts[0])
                design = parts[1]  # LAMOST designation like J041851.70+172316.7
                ra = float(parts[2])
                dec = float(parts[3])
                
                sources.append({
                    'OBSID': obsid,
                    'Design': design,
                    'RA': ra,
                    'Dec': dec
                })
            except:
                pass
        
        df3 = pd.DataFrame(sources)
        
        if len(df3) > 0:
            print(f"\n✓ Successfully extracted {len(df3)} sources")
            
            df3.to_csv(output_dir / 'paper3_lamost_sources.csv', index=False)
            print(f"✓ Saved to: paper3_lamost_sources.csv")
            
            print(f"\nCoordinate ranges:")
            print(f"  RA:  {df3['RA'].min():.2f} to {df3['RA'].max():.2f} degrees")
            print(f"  Dec: {df3['Dec'].min():.2f} to {df3['Dec'].max():.2f} degrees")
            
            print(f"\nFirst 5 sources:")
            print(df3.head().to_string())

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Create master query list for ZTF
print("\n" + "=" * 90)
print("Creating Master ZTF Query List")
print("=" * 90)

try:
    # Combine all sources
    all_sources = []
    
    if 'df2' in locals():
        for idx, row in df2.iterrows():
            all_sources.append({
                'Source': f"NEOWISE_NS_{idx}",
                'RA': row['RA'],
                'Dec': row['Dec'],
                'Paper': 'Neha_Sharma',
                'LCType': row['LCType'],
                'YSO_Class': row['YSO_Class']
            })
    
    if 'df3' in locals():
        for idx, row in df3.iterrows():
            all_sources.append({
                'Source': f"LAMOST_{idx}",
                'RA': row['RA'],
                'Dec': row['Dec'],
                'Paper': 'LAMOST',
                'LCType': 'Unknown',
                'YSO_Class': 'Unknown'
            })
    
    df_all = pd.DataFrame(all_sources)
    
    print(f"\nTotal sources: {len(df_all)}")
    print(f"Paper distribution:")
    for paper in df_all['Paper'].unique():
        count = len(df_all[df_all['Paper'] == paper])
        print(f"  {paper}: {count}")
    
    df_all.to_csv(output_dir / 'ztf_query_master_list.csv', index=False)
    print(f"\n✓ Master list saved to: ztf_query_master_list.csv")
    
    # Create subset for testing (first 50 sources from each paper)
    test_list = []
    for paper in df_all['Paper'].unique():
        paper_sources = df_all[df_all['Paper'] == paper].head(50)
        test_list.append(paper_sources)
    
    df_test = pd.concat(test_list, ignore_index=True)
    df_test.to_csv(output_dir / 'ztf_query_test_list.csv', index=False)
    print(f"✓ Test list (50 per paper) saved to: ztf_query_test_list.csv")

except Exception as e:
    print(f"Error creating master list: {e}")
    import traceback
    traceback.print_exc()

# Print test coordinates matching the user's request
print("\n" + "=" * 90)
print("Test Coordinates (from user specification)")
print("=" * 90)

test_ra = 325.348403
test_dec = 65.927139

print(f"\nTest coordinate: RA = {test_ra}, Dec = {test_dec}")

# Check if this is in our catalogs
if 'df2' in locals():
    nearby_ns = df2[(df2['RA'] - test_ra).abs() < 0.1] & \
                   (df2['Dec'] - test_dec).abs() < 0.1
    if len(nearby_ns) > 0:
        print(f"\nFound {len(nearby_ns)} Paper 2 sources within 0.1 deg:")
        print(nearby_ns.to_string())

if 'df3' in locals():
    nearby_lamost = df3[(df3['RA'] - test_ra).abs() < 0.1] & \
                       (df3['Dec'] - test_dec).abs() < 0.1
    if len(nearby_lamost) > 0:
        print(f"\nFound {len(nearby_lamost)} Paper 3 sources within 0.1 deg:")
        print(nearby_lamost.to_string())

print("\n" + "=" * 90)
print(f"Analysis complete. Files saved to: {output_dir}")
print("=" * 90)
