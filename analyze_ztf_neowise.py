#!/usr/bin/env python3
"""
Script to analyze ZTF lightcurves for NEOWISE YSO sources
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import requests
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting parameters
fsize = 15.5
tsize = 10
tdir = 'in'
major = 3.0
minor = 3.0
style = 'default'

plt.style.use(style)
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor
plt.rcParams["scatter.marker"] = '.'
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = True
plt.rc('font', family='serif')

# Set up output directories
output_dir = Path('/Users/marcus/Desktop/RealYSO/ZTF_Analysis')
output_dir.mkdir(exist_ok=True)
lightcurves_dir = output_dir / 'lightcurves'
plots_dir = output_dir / 'plots'
lightcurves_dir.mkdir(exist_ok=True)
plots_dir.mkdir(exist_ok=True)

def parse_neowise_data():
    """Parse the three NEOWISE data files and extract coordinates"""
    sources = []
    source_info = []
    
    # Paper 2: apjsadc397t2_mrt.txt (Neha & Sharma - Variability)
    print("Parsing Paper 2: apjsadc397t2_mrt.txt (Neha & Sharma)...")
    try:
        df2 = pd.read_csv('/Users/marcus/Desktop/RealYSO/apjsadc397t2_mrt.txt',
                          delim_whitespace=True, skiprows=35)
        print(f"  Found {len(df2)} sources")
        if 'RAdeg' in df2.columns and 'DEdeg' in df2.columns:
            for idx, row in df2.iterrows():
                sources.append([row['RAdeg'], row['DEdeg']])
                source_info.append({
                    'ra': row['RAdeg'],
                    'dec': row['DEdeg'],
                    'paper': 'Neha_Sharma',
                    'name': row.get('Objname', f'NS_{idx}') if 'Objname' in df2.columns else f'NS_{idx}'
                })
            print(f"  ✓ Extracted {len(source_info)} sources with RA/Dec")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Paper 3: apjsadf4e6t4_mrt.txt (LAMOST YSOs)
    print("Parsing Paper 3: apjsadf4e6t4_mrt.txt (LAMOST)...")
    try:
        df3 = pd.read_csv('/Users/marcus/Desktop/RealYSO/apjsadf4e6t4_mrt.txt',
                          delim_whitespace=True, skiprows=31)
        print(f"  Found {len(df3)} sources")
        if 'RAdeg' in df3.columns and 'DEdeg' in df3.columns:
            for idx, row in df3.iterrows():
                sources.append([row['RAdeg'], row['DEdeg']])
                source_info.append({
                    'ra': row['RAdeg'],
                    'dec': row['DEdeg'],
                    'paper': 'LAMOST',
                    'name': row.get('Design', f'LAMOST_{idx}') if 'Design' in df3.columns else f'LAMOST_{idx}'
                })
            print(f"  ✓ Extracted {len(source_info) - len(sources) + len(df3)} sources with RA/Dec")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Paper 1: apjadd25ft1_mrt.txt (FUors - needs sexagesimal conversion)
    print("Parsing Paper 1: apjadd25ft1_mrt.txt (FUors)...")
    try:
        with open('/Users/marcus/Desktop/RealYSO/apjadd25ft1_mrt.txt', 'r') as f:
            lines = f.readlines()
        
        # Parse fixed-width format
        for line in lines[40:]:  # Skip header
            if len(line.strip()) == 0:
                continue
            try:
                # Parse sexagesimal RA: RAh RAm RAs (bytes 131-143)
                rah = int(line[131:133].strip())
                ram = int(line[134:136].strip())
                ras = float(line[137:143].strip())
                ra_deg = rah * 15 + ram * 0.25 + ras * (15/3600)
                
                # Parse sexagesimal Dec: DE- DEd DEm DEs (bytes 145-157)
                de_sign = -1 if line[145] == '-' else 1
                ded = int(line[146:148].strip())
                dem = int(line[149:151].strip())
                des = float(line[152:157].strip())
                dec_deg = de_sign * (ded + dem/60 + des/3600)
                
                sources.append([ra_deg, dec_deg])
                source_info.append({
                    'ra': ra_deg,
                    'dec': dec_deg,
                    'paper': 'FUors',
                    'name': f'FUor_{len([s for s in source_info if s["paper"] == "FUors"])+1}'
                })
            except:
                pass  # Skip malformed lines
        
        n_fuors = len([s for s in source_info if s['paper'] == 'FUors'])
        if n_fuors > 0:
            print(f"  ✓ Extracted {n_fuors} sources with RA/Dec")
    except Exception as e:
        print(f"  Error: {e}")
    
    print(f"\nTotal sources extracted: {len(sources)}")
    print(f"  - Neha & Sharma: {len([s for s in source_info if s['paper'] == 'Neha_Sharma'])}")
    print(f"  - LAMOST: {len([s for s in source_info if s['paper'] == 'LAMOST'])}")
    print(f"  - FUors: {len([s for s in source_info if s['paper'] == 'FUors'])}")
    
    return sources, source_info

def query_ztf_lightcurve(ra, dec, radius=0.000416667):
    """
    Query ZTF lightcurve from IRSA
    radius in degrees (0.000416667 ≈ 1.5 arcsec)
    """
    url = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+{ra}+{dec}+{radius}&BANDNAME=r&NOBS_MIN=10&BAD_CATFLAGS_MASK=32768&FORMAT=csv"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            # Parse the CSV response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            if len(df) > 0 and len(df) > 1:  # More than just header
                return df
    except Exception as e:
        print(f"    Error querying ZTF for ({ra}, {dec}): {e}")
    
    return None

def save_and_analyze_lightcurve(lc, ra, dec, paper_name="unknown"):
    """Save lightcurve and generate analysis"""
    if lc is None or len(lc) == 0:
        return None
    
    # Save CSV
    filename = f"ZTF_{ra:.6f}_{dec:.6f}_r.csv"
    filepath = lightcurves_dir / filename
    lc.to_csv(filepath, index=False)
    
    # Analyze statistics
    if 'mag' in lc.columns and 'magerr' in lc.columns:
        try:
            n_meas = len(lc['mag'])
            mag_std = statistics.stdev(lc['mag'])
            mag_median = statistics.median(lc['mag'])
            err_median = statistics.median(lc['magerr'])
            snr = mag_std / err_median if err_median > 0 else 0
            
            # Create plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.errorbar(lc['mjd'], lc['mag'], yerr=lc['magerr'], 
                       marker='s', markersize=5, color='red', fmt='s', alpha=0.7)
            ax.axhline(mag_median, color='blue', linestyle='-', label=f'Median: {mag_median:.3f}')
            ax.axhline(mag_median + mag_std, color='gray', linestyle='--', alpha=0.7, 
                      label=f'±1σ: {mag_std:.3f}')
            ax.axhline(mag_median - mag_std, color='gray', linestyle='--', alpha=0.7)
            
            ax.set_xlabel('MJD')
            ax.set_ylabel('r [mag]')
            ax.set_title(f'{paper_name}: RA={ra:.4f}, Dec={dec:.4f}')
            ax.invert_yaxis()
            ax.grid(alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            plot_path = plots_dir / f"{filename}.png"
            plt.savefig(plot_path, dpi=100)
            plt.close()
            
            return {
                'filename': filename,
                'ra': ra,
                'dec': dec,
                'n_measurements': n_meas,
                'mag_median': mag_median,
                'mag_std': mag_std,
                'err_median': err_median,
                'snr': snr,
                'paper': paper_name,
                'source_paper': paper_name,
                'source_name': paper_name
            }
        except Exception as e:
            print(f"    Error analyzing lightcurve: {e}")
    
    return None

def test_single_coordinate(ra, dec):
    """Test the notebook functionality with a single coordinate"""
    print(f"\n=== Testing ZTF query with coordinates RA={ra}, Dec={dec} ===")
    lc = query_ztf_lightcurve(ra, dec)
    
    if lc is not None:
        print(f"✓ Successfully retrieved ZTF lightcurve with {len(lc)} measurements")
        stats = save_and_analyze_lightcurve(lc, ra, dec, "Test Coordinate")
        if stats:
            print(f"  - Standard deviation: {stats['mag_std']:.4f} mag")
            print(f"  - Median error: {stats['err_median']:.4f} mag")
            print(f"  - Signal-to-noise ratio: {stats['snr']:.2f}")
            print(f"  - Plot saved to: {plots_dir}/{stats['filename']}.png")
        return True
    else:
        print("✗ No ZTF data found for these coordinates")
        return False

def main():
    print("=" * 80)
    print("NEOWISE - ZTF Lightcurve Analysis")
    print("=" * 80)
    
    # Step 1: Test with the provided coordinates
    test_ra = 325.348403
    test_dec = 65.927139
    test_result = test_single_coordinate(test_ra, test_dec)
    
    if not test_result:
        print("\nWarning: Test query failed. This may indicate network issues.")
        print("Continuing with full analysis anyway...")
    
    # Step 2: Parse NEOWISE data
    print("\n" + "=" * 80)
    print("Parsing NEOWISE sources...")
    print("=" * 80)
    sources, source_info = parse_neowise_data()
    
    if len(sources) == 0:
        print("No sources found in NEOWISE files")
        return
    
    print(f"\nTotal sources to query: {len(sources)}")
    
    # Step 3: Query ZTF for each source
    print("\n" + "=" * 80)
    print("Querying ZTF for NEOWISE sources (this may take a while)...")
    print("=" * 80)
    
    results = []
    found_count = 0
    
    # Query first 100 sources or all if fewer
    n_query = min(100, len(sources))
    for i, (ra, dec) in enumerate(sources[:n_query]):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{n_query}")
        
        lc = query_ztf_lightcurve(ra, dec)
        if lc is not None and len(lc) > 1:
            source = source_info[i] if i < len(source_info) else {'paper': 'Unknown', 'name': f'Source_{i}'}
            stats = save_and_analyze_lightcurve(lc, ra, dec, f"{source['paper']}_{source['name']}")
            if stats:
                stats['source_name'] = source['name']
                stats['source_paper'] = source['paper']
                results.append(stats)
                found_count += 1
    
    # Step 4: Summary report
    print("\n" + "=" * 80)
    print("Analysis Summary")
    print("=" * 80)
    print(f"Total sources in NEOWISE catalogs: {len(sources)}")
    print(f"Sources queried: {n_query}")
    print(f"Sources with ZTF lightcurves: {found_count}")
    print(f"Detection rate: {100*found_count/n_query:.1f}%")
    
    if found_count > 0:
        results_df = pd.DataFrame(results)
        print(f"\nOptical variability statistics (for detected sources):")
        print(f"  Mean magnitude std dev: {results_df['mag_std'].mean():.4f} mag")
        print(f"  Median magnitude std dev: {results_df['mag_std'].median():.4f} mag")
        print(f"  Max magnitude std dev: {results_df['mag_std'].max():.4f} mag")
        print(f"  Min magnitude std dev: {results_df['mag_std'].min():.4f} mag")
        print(f"  Mean SNR: {results_df['snr'].mean():.2f}")
        print(f"  Mean measurements per source: {results_df['n_measurements'].mean():.0f}")
        
        # Save results to CSV
        results_csv = output_dir / 'ztf_neowise_sources.csv'
        results_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to: {results_csv}")
        print(f"Lightcurves saved to: {lightcurves_dir}")
        print(f"Plots saved to: {plots_dir}")
        
        # Print summary by paper
        print(f"\nDetections by paper:")
        for paper in results_df['source_paper'].unique():
            n = len(results_df[results_df['source_paper'] == paper])
            print(f"  {paper}: {n} sources")
    else:
        print("\nNo sources with ZTF data found in the queried sample")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
