#!/usr/bin/env python3
"""
ZTF Lightcurve Validation Script
Replicates the notebook functionality and validates with test coordinates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import requests
from io import StringIO
import time
from pathlib import Path

# Setup output directory
output_dir = Path('/Users/marcus/Desktop/RealYSO/ZTF_Validation')
output_dir.mkdir(exist_ok=True)

# Plotting setup (matching notebook)
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

print("=" * 90)
print("ZTF LIGHTCURVE VALIDATION & ANALYSIS")
print("=" * 90)

def query_ztf_irsa(ra, dec, radius=0.000416667, band='r', max_retries=3, timeout=60):
    """
    Query IRSA ZTF database for lightcurve
    
    Parameters:
    -----------
    ra, dec : float
        Coordinates in decimal degrees
    radius : float
        Search radius in degrees (default 1.5 arcsec)
    band : str
        Band name ('r' or 'g')
    max_retries : int
        Number of retries for failed queries
    timeout : int
        Timeout in seconds
    
    Returns:
    --------
    pd.DataFrame or None
        Lightcurve data if found, None otherwise
    """
    
    url = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+{ra}+{dec}+{radius}&BANDNAME={band}&NOBS_MIN=10&BAD_CATFLAGS_MASK=32768&FORMAT=csv"
    
    for attempt in range(max_retries):
        try:
            print(f"  Query attempt {attempt+1}/{max_retries}...", end='', flush=True)
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                # Parse CSV
                df = pd.read_csv(StringIO(response.text))
                if len(df) > 1:  # More than just header row
                    print(f" ✓ Success! Found {len(df)} measurements")
                    return df
                else:
                    print(f" - No data found")
                    return None
            else:
                print(f" - HTTP {response.status_code}")
        
        except requests.Timeout:
            print(f" ✗ Timeout after {timeout}s", end='')
            if attempt < max_retries - 1:
                print(", retrying...")
                time.sleep(2)
            else:
                print()
        except Exception as e:
            print(f" ✗ Error: {e}")
    
    return None

def analyze_lightcurve(lc_df, ra, dec, source_name="Unknown"):
    """
    Calculate statistics and return analysis results
    
    Returns:
    --------
    dict with keys: n_measurements, mag_std, mag_median, mag_err_median, snr
    """
    
    if lc_df is None or len(lc_df) == 0:
        return None
    
    try:
        mag = lc_df.get('mag', lc_df.get('magnitude', None))
        magerr = lc_df.get('magerr', lc_df.get('magerr', None))
        
        if mag is None or magerr is None:
            print(f"    Warning: Could not find magnitude columns. Available: {lc_df.columns.tolist()}")
            return None
        
        # Filter out bad measurements
        valid = ~(mag.isna() | magerr.isna())
        mag_clean = mag[valid]
        magerr_clean = magerr[valid]
        
        if len(mag_clean) < 3:
            return None
        
        n_meas = len(mag_clean)
        mag_std = statistics.stdev(mag_clean)
        mag_median = statistics.median(mag_clean)
        mag_err_median = statistics.median(magerr_clean)
        snr = mag_std / mag_err_median if mag_err_median > 0 else 0
        
        return {
            'source': source_name,
            'ra': ra,
            'dec': dec,
            'n_measurements': n_meas,
            'mag_median': mag_median,
            'mag_std': mag_std,
            'mag_err_median': mag_err_median,
            'snr': snr,
            'quality': 'high' if snr > 5 else ('medium' if snr > 2 else 'low')
        }
    
    except Exception as e:
        print(f"    Error analyzing lightcurve: {e}")
        return None

def plot_lightcurve(lc_df, ra, dec, source_name="Unknown", output_path=None):
    """
    Create lightcurve plot
    """
    
    if lc_df is None or len(lc_df) == 0:
        return False
    
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        mjd = lc_df.get('mjd', lc_df.get('jd', None))
        mag = lc_df.get('mag', lc_df.get('magnitude', None))
        magerr = lc_df.get('magerr', None)
        
        if mjd is None or mag is None:
            return False
        
        # Filter valid data
        valid = ~(mag.isna() | magerr.isna())
        mjd_clean = mjd[valid]
        mag_clean = mag[valid]
        magerr_clean = magerr[valid]
        
        # Plot
        ax.errorbar(mjd_clean, mag_clean, yerr=magerr_clean, 
                   marker='s', markersize=4, color='red', linestyle='none', 
                   elinewidth=1, capsize=3, alpha=0.7, label='ZTF r-band')
        
        # Add median line
        mag_med = statistics.median(mag_clean)
        ax.axhline(mag_med, color='blue', linestyle='-', linewidth=2, label=f'Median: {mag_med:.2f}')
        
        # Add std dev lines
        mag_std = statistics.stdev(mag_clean) if len(mag_clean) > 1 else 0
        ax.axhline(mag_med + mag_std, color='gray', linestyle='--', alpha=0.5, label=f'±1σ: {mag_std:.3f}')
        ax.axhline(mag_med - mag_std, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('MJD (days)', fontsize=12)
        ax.set_ylabel('Magnitude (r-band)', fontsize=12)
        ax.set_title(f'{source_name}\nRA={ra:.4f}°, Dec={dec:.4f}°', fontsize=13)
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=100)
            print(f"    ✓ Plot saved: {output_path.name}")
        
        plt.close()
        return True
    
    except Exception as e:
        print(f"    Error creating plot: {e}")
        return False

# PHASE 1: TEST WITH PROVIDED COORDINATES
print("\n" + "=" * 90)
print("PHASE 1: NOTEBOOK VALIDATION")
print("Testing with provided coordinates: RA=325.348403°, Dec=+65.927139°")
print("=" * 90)

test_ra = 325.348403
test_dec = 65.927139

print(f"\nQuerying ZTF for test coordinate...")
test_lc = query_ztf_irsa(test_ra, test_dec, timeout=60)

if test_lc is not None:
    print(f"\n✓ SUCCESS: ZTF query returned data")
    print(f"  Lightcurve has {len(test_lc)} observations")
    print(f"  Columns: {test_lc.columns.tolist()[:8]}...")
    
    # Analyze
    test_stats = analyze_lightcurve(test_lc, test_ra, test_dec, "Test_Coordinate")
    if test_stats:
        print(f"\n  Lightcurve Statistics:")
        print(f"    Measurements:     {test_stats['n_measurements']}")
        print(f"    Median magnitude: {test_stats['mag_median']:.3f} mag")
        print(f"    Std deviation:    {test_stats['mag_std']:.4f} mag")
        print(f"    Median error:     {test_stats['mag_err_median']:.4f} mag")
        print(f"    SNR (var/err):    {test_stats['snr']:.2f}")
        print(f"    Quality:          {test_stats['quality'].upper()}")
        
        # Plot
        print(f"\n  Generating lightcurve plot...")
        plot_path = output_dir / f"test_coordinate_{test_ra}_{test_dec}.png"
        if plot_lightcurve(test_lc, test_ra, test_dec, "Test Coordinate (Notebook Validation)", plot_path):
            print(f"    ✓ Saved to: ZTF_Validation/")
else:
    print(f"\n✗ FAILED: No ZTF data for test coordinate")
    print(f"  This may indicate:")
    print(f"    1. Coordinate is outside ZTF survey area")
    print(f"    2. No sources with ≥10 observations at this location")
    print(f"    3. Network/API timeout issues")

# PHASE 2: QUERY NEOWISE SOURCES
print("\n" + "=" * 90)
print("PHASE 2: ZTF QUERIES FOR NEOWISE SOURCES")
print("=" * 90)

# Load NEOWISE variable sources
paper2_path = Path('/Users/marcus/Desktop/RealYSO/Analysis/paper2_variable_sources.csv')
paper3_path = Path('/Users/marcus/Desktop/RealYSO/Analysis/paper3_lamost_sources.csv')

all_results = []

# Paper 2: NEOWISE Variable Sources
if paper2_path.exists():
    print(f"\nLoading Paper 2 (NEOWISE Variable) sources...")
    df2 = pd.read_csv(paper2_path)
    print(f"  Loaded {len(df2)} sources")
    
    # Test with first 10 variable sources
    print(f"\nQuerying first 10 NEOWISE variable sources...")
    n_detected = 0
    
    for i, row in df2.head(10).iterrows():
        source_name = f"NS_Variable_{i}_{row['Name']}"
        print(f"\n  [{i+1}/10] {row['Name']} ({row['LCType']}) - RA={row['RA']:.4f}, Dec={row['Dec']:.4f}")
        
        lc = query_ztf_irsa(row['RA'], row['Dec'], timeout=60)
        
        if lc is not None and len(lc) > 1:
            stats = analyze_lightcurve(lc, row['RA'], row['Dec'], source_name)
            if stats:
                stats['paper'] = 'NEOWISE'
                stats['lctype'] = row['LCType']
                stats['yso_class'] = row['YSO_Class']
                all_results.append(stats)
                n_detected += 1
                
                # Plot
                plot_path = output_dir / f"{source_name}.png"
                plot_lightcurve(lc, row['RA'], row['Dec'], 
                              f"{row['Name']} ({row['LCType']})", plot_path)
    
    print(f"\n  Paper 2 Summary: {n_detected}/10 sources detected in ZTF")

# Paper 3: LAMOST Sources
if paper3_path.exists():
    print(f"\nLoading Paper 3 (LAMOST H-alpha) sources...")
    df3 = pd.read_csv(paper3_path)
    print(f"  Loaded {len(df3)} sources")
    
    # Test with first 10 H-alpha sources
    print(f"\nQuerying first 10 LAMOST H-alpha sources...")
    n_detected = 0
    
    for i, row in df3.head(10).iterrows():
        source_name = f"LAMOST_{i}_{row['Design']}"
        print(f"\n  [{i+1}/10] {row['Design']} - RA={row['RA']:.4f}, Dec={row['Dec']:.4f}")
        
        lc = query_ztf_irsa(row['RA'], row['Dec'], timeout=60)
        
        if lc is not None and len(lc) > 1:
            stats = analyze_lightcurve(lc, row['RA'], row['Dec'], source_name)
            if stats:
                stats['paper'] = 'LAMOST'
                stats['lctype'] = 'H-alpha_emission'
                all_results.append(stats)
                n_detected += 1
                
                # Plot
                plot_path = output_dir / f"{source_name}.png"
                plot_lightcurve(lc, row['RA'], row['Dec'], 
                              f"{row['Design']} (H-alpha)", plot_path)
    
    print(f"\n  Paper 3 Summary: {n_detected}/10 sources detected in ZTF")

# SUMMARY
print("\n" + "=" * 90)
print("VALIDATION RESULTS SUMMARY")
print("=" * 90)

if len(all_results) > 0:
    results_df = pd.DataFrame(all_results)
    
    print(f"\nSuccessfully detected and analyzed: {len(results_df)} sources in ZTF")
    
    print(f"\nData Quality Distribution:")
    print(results_df['quality'].value_counts())
    
    print(f"\nVariability Statistics:")
    print(f"  Mean SNR:              {results_df['snr'].mean():.2f}")
    print(f"  Median std deviation:  {results_df['mag_std'].median():.4f} mag")
    print(f"  Mean measurements:     {results_df['n_measurements'].mean():.0f}")
    
    print(f"\nDetailed Results:")
    print(results_df[['source', 'n_measurements', 'mag_std', 'snr', 'quality']].to_string())
    
    # Save results
    results_df.to_csv(output_dir / 'ztf_validation_results.csv', index=False)
    print(f"\n✓ Results saved to: ZTF_Validation/ztf_validation_results.csv")
else:
    print(f"\n⚠ No sources successfully detected in ZTF")
    print(f"  This may indicate timeout issues with the IRSA API")
    print(f"  Recommendation: Use batch queries or alternative methods")

print(f"\n✓ All outputs saved to: {output_dir}/")
print("\n" + "=" * 90)
