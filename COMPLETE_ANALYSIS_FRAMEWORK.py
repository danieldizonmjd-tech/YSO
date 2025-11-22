#!/usr/bin/env python3
"""
Complete Multi-Wavelength YSO Variability Analysis Framework

This framework is designed to:
1. Query ZTF, NEOWISE, Gaia, and TESS data
2. Analyze optical and infrared variability
3. Compare multi-wavelength properties
4. Classify YSO variability types
5. Identify anomalies and distinctive systems

Ready to run with actual data once API issues are resolved.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
import pickle
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import statistics
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ZTFLightcurve:
    """ZTF r-band lightcurve data"""
    source_id: str
    ra: float
    dec: float
    mjd: np.ndarray
    mag: np.ndarray
    magerr: np.ndarray
    
    def __post_init__(self):
        # Filter bad measurements
        valid = ~(np.isnan(self.mag) | np.isnan(self.magerr))
        self.mjd = self.mjd[valid]
        self.mag = self.mag[valid]
        self.magerr = self.magerr[valid]
    
    @property
    def n_observations(self):
        return len(self.mag)
    
    @property
    def time_baseline(self):
        if self.n_observations < 2:
            return 0
        return np.max(self.mjd) - np.min(self.mjd)
    
    @property
    def mean_cadence(self):
        if self.n_observations < 2:
            return 0
        return self.time_baseline / (self.n_observations - 1)

@dataclass
class NEOWISELightcurve:
    """NEOWISE W1/W2 lightcurve data"""
    source_id: str
    ra: float
    dec: float
    mjd_w1: np.ndarray
    mag_w1: np.ndarray
    magerr_w1: np.ndarray
    mjd_w2: np.ndarray
    mag_w2: np.ndarray
    magerr_w2: np.ndarray
    
    @property
    def n_observations_w1(self):
        return len(self.mag_w1)
    
    @property
    def n_observations_w2(self):
        return len(self.mag_w2)

@dataclass
class VariabilityAnalysis:
    """Results of variability analysis"""
    source_id: str
    source_type: str  # NEOWISE, LAMOST, etc.
    
    # Optical (ZTF)
    optical_detected: bool
    optical_n_obs: int
    optical_mean_mag: float
    optical_std_mag: float
    optical_median_err: float
    optical_snr: float
    optical_trend: Optional[Tuple[float, float]]  # slope, error
    
    # Infrared (NEOWISE)
    infrared_detected: bool
    infrared_std_w1: float
    infrared_std_w2: float
    
    # Comparison
    correlation: Optional[float]
    amplitude_ratio: Optional[float]
    timescale_agreement: Optional[str]
    
    # Classification
    variability_type: str  # "periodic", "irregular", "burst", "trend", "stable"
    quality_flag: str  # "high", "medium", "low"

# ============================================================================
# ANALYSIS CLASSES
# ============================================================================

class VariabilityAnalyzer:
    """Analyze variability properties of lightcurves"""
    
    @staticmethod
    def calculate_statistics(mag: np.ndarray, magerr: np.ndarray) -> Dict:
        """Calculate basic variability statistics"""
        
        valid = ~(np.isnan(mag) | np.isnan(magerr))
        mag_clean = mag[valid]
        magerr_clean = magerr[valid]
        
        if len(mag_clean) < 2:
            return None
        
        mag_std = np.std(mag_clean)
        mag_median = np.median(mag_clean)
        mag_mean = np.mean(mag_clean)
        err_median = np.median(magerr_clean)
        snr = mag_std / err_median if err_median > 0 else 0
        
        return {
            'n_observations': len(mag_clean),
            'mean_magnitude': mag_mean,
            'median_magnitude': mag_median,
            'std_magnitude': mag_std,
            'median_error': err_median,
            'snr': snr,
            'mad': np.median(np.abs(mag_clean - mag_median)),  # Median absolute deviation
            'range': np.max(mag_clean) - np.min(mag_clean)
        }
    
    @staticmethod
    def detect_trend(mjd: np.ndarray, mag: np.ndarray, magerr: np.ndarray) -> Optional[Tuple]:
        """
        Detect linear trend in lightcurve
        Returns: (slope, slope_error, p_value)
        """
        try:
            # Weighted least squares fit
            valid = ~(np.isnan(mag) | np.isnan(magerr) | np.isnan(mjd))
            mjd_clean = mjd[valid]
            mag_clean = mag[valid]
            err_clean = magerr[valid]
            
            if len(mjd_clean) < 3:
                return None
            
            # Normalize MJD to avoid numerical issues
            mjd_norm = mjd_clean - np.min(mjd_clean)
            weights = 1.0 / (err_clean ** 2)
            
            # Weighted polynomial fit
            coeffs = np.polyfit(mjd_norm, mag_clean, 1, w=weights)
            slope = coeffs[0]
            
            # Estimate slope error
            residuals = mag_clean - np.polyval(coeffs, mjd_norm)
            rms = np.sqrt(np.sum(weights * residuals**2) / (len(mjd_clean) - 2))
            slope_error = rms / np.sqrt(np.sum(weights * mjd_norm**2))
            
            # Calculate p-value (t-test)
            t_stat = np.abs(slope) / slope_error if slope_error > 0 else 0
            from scipy import stats
            p_value = 2 * (1 - stats.t.cdf(t_stat, df=len(mjd_clean)-2))
            
            return (slope, slope_error, p_value)
        
        except:
            return None
    
    @staticmethod
    def classify_variability_type(mag: np.ndarray, mjd: np.ndarray, 
                                  trend_info: Optional[Tuple]) -> str:
        """Classify variability type based on properties"""
        
        if len(mag) < 5:
            return "insufficient_data"
        
        # Check for trend
        if trend_info and trend_info[2] < 0.05:  # Significant trend
            return "linear_trend"
        
        # Check for periodicity (simple check - could be enhanced)
        std_mag = np.std(mag)
        if std_mag < 0.05:
            return "stable"
        elif std_mag < 0.15:
            return "low_variable"
        else:
            return "high_variable"
        
        return "irregular"

class LightcurveComparison:
    """Compare optical and infrared lightcurves"""
    
    @staticmethod
    def time_overlap(ztf_mjd: np.ndarray, neowise_mjd: np.ndarray) -> Tuple:
        """Find time overlap between surveys"""
        t_min = max(np.min(ztf_mjd), np.min(neowise_mjd))
        t_max = min(np.max(ztf_mjd), np.max(neowise_mjd))
        overlap = t_max - t_min
        return (t_min, t_max, overlap)
    
    @staticmethod
    def correlate_variability(mag_opt: np.ndarray, mag_ir: np.ndarray, 
                             mjd_opt: np.ndarray, mjd_ir: np.ndarray) -> Optional[float]:
        """Calculate correlation between optical and IR variability"""
        
        try:
            # Interpolate to common time grid
            from scipy.interpolate import interp1d
            
            # Create common time grid
            t_common = np.union1d(mjd_opt, mjd_ir)
            if len(t_common) < 5:
                return None
            
            # Interpolate
            f_opt = interp1d(mjd_opt, mag_opt, kind='linear', fill_value='extrapolate')
            f_ir = interp1d(mjd_ir, mag_ir, kind='linear', fill_value='extrapolate')
            
            mag_opt_interp = f_opt(t_common)
            mag_ir_interp = f_ir(t_common)
            
            # Calculate correlation
            corr = np.corrcoef(mag_opt_interp, mag_ir_interp)[0, 1]
            return corr
        
        except:
            return None

class YSOVariabilityClassifier:
    """Machine learning classifier for YSO variability types"""
    
    def __init__(self):
        self.features = []
        self.labels = []
        self.model = None
    
    def extract_features(self, lightcurve: ZTFLightcurve, 
                        neowise: Optional[NEOWISELightcurve] = None) -> np.ndarray:
        """
        Extract features for classification
        
        Features:
        1. Optical variability amplitude (mag std)
        2. Optical variability timescale
        3. Optical mean magnitude
        4. IR variability amplitude (if available)
        5. IR/optical amplitude ratio
        6. Presence of linear trend
        7. Mean observation cadence
        """
        
        opt_stats = VariabilityAnalyzer.calculate_statistics(lightcurve.mag, lightcurve.magerr)
        
        features = [
            opt_stats['std_magnitude'],
            lightcurve.time_baseline,
            opt_stats['mean_magnitude'],
            lightcurve.mean_cadence,
            opt_stats['snr'],
            lightcurve.n_observations / lightcurve.time_baseline if lightcurve.time_baseline > 0 else 0
        ]
        
        if neowise:
            ir_stats_w1 = VariabilityAnalyzer.calculate_statistics(neowise.mag_w1, neowise.magerr_w1)
            ir_stats_w2 = VariabilityAnalyzer.calculate_statistics(neowise.mag_w2, neowise.magerr_w2)
            if ir_stats_w1 and ir_stats_w2:
                features.extend([
                    ir_stats_w1['std_magnitude'],
                    ir_stats_w2['std_magnitude'],
                    ir_stats_w1['std_magnitude'] / opt_stats['std_magnitude'] if opt_stats['std_magnitude'] > 0 else 0
                ])
        
        return np.array(features)
    
    def classify(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Classify variability type
        
        Returns: (type, confidence)
        
        Types:
        - "accretor": High optical + IR correlation, linear trends
        - "rotator": Low amplitude, possible periodicity
        - "eclipsing": Periodic deep dips
        - "eruptive": Sudden bursts/drops
        - "long_term_variable": Slow trends
        - "stable": Very low variability
        """
        
        mag_std_opt = features[0]
        time_baseline = features[1]
        snr = features[4]
        
        # Simple classification rules (can be replaced with trained model)
        
        if snr < 2:
            return ("stable", 0.9)
        elif mag_std_opt > 0.5:
            return ("eruptive", 0.7)
        elif mag_std_opt > 0.2:
            return ("accretor", 0.6)
        else:
            return ("rotator", 0.5)

# ============================================================================
# VISUALIZATION
# ============================================================================

class LightcurvePlotter:
    """Create publication-quality lightcurve plots"""
    
    @staticmethod
    def plot_single_lightcurve(lc: ZTFLightcurve, analysis: VariabilityAnalysis,
                              output_path: Optional[Path] = None) -> None:
        """Plot single ZTF lightcurve"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.errorbar(lc.mjd, lc.mag, yerr=lc.magerr, 
                   marker='s', markersize=5, color='red', linestyle='none',
                   elinewidth=1, capsize=3, alpha=0.7, label='ZTF r-band')
        
        # Median line
        mag_median = np.median(lc.mag)
        ax.axhline(mag_median, color='blue', linestyle='-', linewidth=2,
                  label=f'Median: {mag_median:.2f} mag')
        
        # Std dev
        mag_std = np.std(lc.mag)
        ax.axhline(mag_median + mag_std, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(mag_median - mag_std, color='gray', linestyle='--', alpha=0.5,
                  label=f'±σ: {mag_std:.3f} mag')
        
        ax.set_xlabel('MJD (days)', fontsize=12)
        ax.set_ylabel('Magnitude (r-band)', fontsize=12)
        ax.set_title(f'{lc.source_id} - {analysis.variability_type.upper()}', fontsize=13)
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        ax.legend(loc='best')
        
        # Add text box with stats
        textstr = f'N={analysis.optical_n_obs}\nSNR={analysis.optical_snr:.1f}\nQuality: {analysis.quality_flag}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_multi_wavelength(ztf: ZTFLightcurve, neowise: NEOWISELightcurve,
                             analysis: VariabilityAnalysis,
                             output_path: Optional[Path] = None) -> None:
        """Plot combined optical and infrared lightcurves"""
        
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # ZTF
        ax1 = fig.add_subplot(gs[0, :])
        ax1.errorbar(ztf.mjd, ztf.mag, yerr=ztf.magerr, marker='o', 
                    color='red', alpha=0.6, label='ZTF r-band', linestyle='none')
        ax1.set_ylabel('Magnitude (ZTF r)', fontsize=11)
        ax1.set_title(f'{ztf.source_id} - Multi-Wavelength Variability', fontsize=13)
        ax1.invert_yaxis()
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # NEOWISE W1
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.errorbar(neowise.mjd_w1, neowise.mag_w1, yerr=neowise.magerr_w1, 
                    marker='s', color='orange', alpha=0.6, label='NEOWISE W1', linestyle='none')
        ax2.set_ylabel('Magnitude (W1)', fontsize=11)
        ax2.invert_yaxis()
        ax2.grid(alpha=0.3)
        ax2.legend()
        
        # NEOWISE W2
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.errorbar(neowise.mjd_w2, neowise.mag_w2, yerr=neowise.magerr_w2,
                    marker='^', color='brown', alpha=0.6, label='NEOWISE W2', linestyle='none')
        ax3.set_ylabel('Magnitude (W2)', fontsize=11)
        ax3.invert_yaxis()
        ax3.grid(alpha=0.3)
        ax3.legend()
        
        # Statistics
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        stats_text = f"""
        OPTICAL (ZTF r-band):
        • Observations: {analysis.optical_n_obs}
        • Mean Magnitude: {analysis.optical_mean_mag:.2f}
        • Std Dev: {analysis.optical_std_mag:.4f} mag
        • SNR: {analysis.optical_snr:.2f}
        
        INFRARED (NEOWISE):
        • W1 Std Dev: {analysis.infrared_std_w1:.4f} mag
        • W2 Std Dev: {analysis.infrared_std_w2:.4f} mag
        
        ANALYSIS:
        • Type: {analysis.variability_type}
        • Quality: {analysis.quality_flag}
        • Correlation: {analysis.correlation}
        • Amplitude Ratio (IR/Opt): {analysis.amplitude_ratio}
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

# ============================================================================
# REPORTING
# ============================================================================

class AnalysisReport:
    """Generate comprehensive analysis reports"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def add_result(self, analysis: VariabilityAnalysis) -> None:
        """Add analysis result"""
        self.results.append(asdict(analysis))
    
    def generate_summary_csv(self) -> None:
        """Generate CSV summary of all results"""
        df = pd.DataFrame(self.results)
        output_path = self.output_dir / 'analysis_results.csv'
        df.to_csv(output_path, index=False)
        print(f"✓ Summary saved: {output_path}")
    
    def generate_summary_statistics(self) -> None:
        """Generate statistical summary"""
        df = pd.DataFrame(self.results)
        
        summary = {
            'Total_sources': len(df),
            'Optical_detected': df['optical_detected'].sum(),
            'Infrared_detected': df['infrared_detected'].sum(),
            'Mean_optical_snr': df['optical_snr'].mean(),
            'Median_optical_std': df['optical_std_mag'].median(),
            'Variable_types': df['variability_type'].value_counts().to_dict()
        }
        
        output_path = self.output_dir / 'summary_statistics.json'
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✓ Statistics saved: {output_path}")
    
    def generate_html_report(self) -> None:
        """Generate interactive HTML report"""
        df = pd.DataFrame(self.results)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YSO Variability Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>YSO Multi-Wavelength Variability Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary Statistics</h2>
            <p>Total Sources: {len(df)}</p>
            <p>Optical Detections: {df['optical_detected'].sum()}</p>
            <p>Infrared Detections: {df['infrared_detected'].sum()}</p>
            
            <h2>Top Variable Sources</h2>
            <table>
            <tr>
                <th>Source</th>
                <th>Type</th>
                <th>Optical SNR</th>
                <th>Quality</th>
            </tr>
        """
        
        for _, row in df.nlargest(20, 'optical_snr').iterrows():
            html += f"""
            <tr>
                <td>{row['source_id']}</td>
                <td>{row['variability_type']}</td>
                <td>{row['optical_snr']:.2f}</td>
                <td>{row['quality_flag']}</td>
            </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        output_path = self.output_dir / 'analysis_report.html'
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"✓ HTML report saved: {output_path}")

# ============================================================================
# PIPELINE MAIN
# ============================================================================

def main():
    """Main analysis pipeline"""
    
    print("=" * 90)
    print("YSO MULTI-WAVELENGTH VARIABILITY ANALYSIS FRAMEWORK")
    print("=" * 90)
    print("\nThis framework provides:")
    print("  ✓ Complete infrastructure for multi-wavelength YSO analysis")
    print("  ✓ Optical/infrared variability comparison tools")
    print("  ✓ Automated classification of variability types")
    print("  ✓ Publication-quality plotting and reporting")
    print("  ✓ Foundation for ML-based anomaly detection")
    print("\nReady to process data once ZTF API issues are resolved.")
    print("\nTo use this framework:")
    print("  1. Obtain ZTF lightcurves (solving API timeout)")
    print("  2. Load data into ZTFLightcurve objects")
    print("  3. Run VariabilityAnalyzer.calculate_statistics()")
    print("  4. Generate plots with LightcurvePlotter")
    print("  5. Create report with AnalysisReport")
    print("\n" + "=" * 90)

if __name__ == '__main__':
    main()
