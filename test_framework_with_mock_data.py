#!/usr/bin/env python3
"""
Comprehensive Test Suite for COMPLETE_ANALYSIS_FRAMEWORK

Tests all major components using mock data:
- VariabilityAnalyzer (statistics, trend detection, classification)
- LightcurveComparison (multi-wavelength correlation)
- YSOVariabilityClassifier (feature extraction and classification)
- LightcurvePlotter (visualization)
- AnalysisReport (output generation)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
import sys

from COMPLETE_ANALYSIS_FRAMEWORK import (
    ZTFLightcurve, NEOWISELightcurve, VariabilityAnalysis,
    VariabilityAnalyzer, LightcurveComparison, YSOVariabilityClassifier,
    LightcurvePlotter, AnalysisReport
)
from mock_data_generator import MockVariabilityGenerator, RealisticYSODataset


class FrameworkTester:
    """Comprehensive testing suite for the analysis framework"""
    
    def __init__(self, output_dir: str = './framework_test_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.test_log = []
    
    def log(self, message: str, level: str = 'INFO'):
        """Log test messages"""
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] [{level}] {message}"
        self.test_log.append(log_msg)
        print(log_msg)
    
    def test_variability_analyzer(self) -> bool:
        """Test VariabilityAnalyzer component"""
        self.log("=" * 80)
        self.log("TEST 1: VariabilityAnalyzer")
        self.log("=" * 80)
        
        try:
            var_types = ['stable', 'periodic', 'irregular', 'burst', 'linear_trend']
            
            for var_type in var_types:
                self.log(f"\n  Testing {var_type} variability...", 'TEST')
                
                ztf = MockVariabilityGenerator.generate_ztf_lightcurve(
                    f'TEST_{var_type}', 325.5, 65.9,
                    variability_type=var_type,
                    n_obs=50,
                    amplitude=0.3
                )
                
                stats = VariabilityAnalyzer.calculate_statistics(ztf.mag, ztf.magerr)
                
                self.log(f"    • n_obs: {stats['n_observations']}")
                self.log(f"    • mean_mag: {stats['mean_magnitude']:.3f}")
                self.log(f"    • std_mag: {stats['std_magnitude']:.4f}")
                self.log(f"    • SNR: {stats['snr']:.2f}")
                self.log(f"    • MAD: {stats['mad']:.4f}")
                self.log(f"    • Range: {stats['range']:.3f} mag")
                
                trend = VariabilityAnalyzer.detect_trend(ztf.mjd, ztf.mag, ztf.magerr)
                if trend:
                    self.log(f"    • Trend: slope={trend[0]:.6f} ± {trend[1]:.6f}, p={trend[2]:.4f}")
                else:
                    self.log(f"    • Trend: None detected")
                
                var_class = VariabilityAnalyzer.classify_variability_type(
                    ztf.mag, ztf.mjd, trend
                )
                self.log(f"    • Classification: {var_class}")
            
            self.log("\n✓ VariabilityAnalyzer tests PASSED", 'SUCCESS')
            return True
        
        except Exception as e:
            self.log(f"✗ VariabilityAnalyzer tests FAILED: {e}", 'ERROR')
            return False
    
    def test_lightcurve_comparison(self) -> bool:
        """Test LightcurveComparison component"""
        self.log("\n" + "=" * 80)
        self.log("TEST 2: LightcurveComparison")
        self.log("=" * 80)
        
        try:
            self.log("\n  Generating multi-wavelength lightcurves...", 'TEST')
            
            ztf = MockVariabilityGenerator.generate_ztf_lightcurve(
                'MW_TEST', 325.5, 65.9, 'irregular', n_obs=80, amplitude=0.35
            )
            neowise = MockVariabilityGenerator.generate_neowise_lightcurve(
                'MW_TEST', 325.5, 65.9, amplitude_w1=0.25, amplitude_w2=0.20, n_obs=60
            )
            
            self.log(f"    • ZTF: {ztf.n_observations} observations, {ztf.time_baseline:.1f} day baseline")
            self.log(f"    • NEOWISE W1: {neowise.n_observations_w1} observations")
            self.log(f"    • NEOWISE W2: {neowise.n_observations_w2} observations")
            
            overlap = LightcurveComparison.time_overlap(ztf.mjd, neowise.mjd_w1)
            self.log(f"\n  Time overlap analysis:")
            self.log(f"    • Overlap range: {overlap[0]:.1f} - {overlap[1]:.1f}")
            self.log(f"    • Duration: {overlap[2]:.1f} days")
            
            corr = LightcurveComparison.correlate_variability(
                ztf.mag, neowise.mag_w1, ztf.mjd, neowise.mjd_w1
            )
            
            if corr is not None:
                self.log(f"\n  Variability correlation:")
                self.log(f"    • Optical-IR correlation: {corr:.3f}")
            else:
                self.log(f"    • Correlation: Could not compute")
            
            self.log("\n✓ LightcurveComparison tests PASSED", 'SUCCESS')
            return True
        
        except Exception as e:
            self.log(f"✗ LightcurveComparison tests FAILED: {e}", 'ERROR')
            return False
    
    def test_classifier_features(self) -> bool:
        """Test YSOVariabilityClassifier feature extraction"""
        self.log("\n" + "=" * 80)
        self.log("TEST 3: YSOVariabilityClassifier - Feature Extraction")
        self.log("=" * 80)
        
        try:
            classifier = YSOVariabilityClassifier()
            
            self.log("\n  Extracting features from mock lightcurves...", 'TEST')
            
            for var_type in ['stable', 'irregular', 'burst']:
                ztf = MockVariabilityGenerator.generate_ztf_lightcurve(
                    f'FEAT_{var_type}', 325.5, 65.9,
                    variability_type=var_type, n_obs=60, amplitude=0.25
                )
                neowise = MockVariabilityGenerator.generate_neowise_lightcurve(
                    f'FEAT_{var_type}', 325.5, 65.9
                )
                
                features = classifier.extract_features(ztf, neowise)
                
                self.log(f"\n  {var_type.upper()} - Extracted {len(features)} features:")
                feature_names = [
                    'opt_std', 'time_baseline', 'opt_mean_mag', 'cadence',
                    'opt_snr', 'obs_rate', 'ir_w1_std', 'ir_w2_std', 'ir_opt_ratio'
                ]
                for i, (name, val) in enumerate(zip(feature_names[:len(features)], features)):
                    self.log(f"    • {name}: {val:.4f}")
            
            self.log("\n✓ Feature extraction tests PASSED", 'SUCCESS')
            return True
        
        except Exception as e:
            self.log(f"✗ Feature extraction tests FAILED: {e}", 'ERROR')
            return False
    
    def test_classifier_prediction(self) -> bool:
        """Test YSOVariabilityClassifier predictions"""
        self.log("\n" + "=" * 80)
        self.log("TEST 4: YSOVariabilityClassifier - Predictions")
        self.log("=" * 80)
        
        try:
            classifier = YSOVariabilityClassifier()
            
            self.log("\n  Testing classification on diverse lightcurves...", 'TEST')
            
            predictions = {}
            for var_type in ['stable', 'periodic', 'irregular', 'burst', 'linear_trend']:
                ztf = MockVariabilityGenerator.generate_ztf_lightcurve(
                    f'PRED_{var_type}', 325.5, 65.9,
                    variability_type=var_type, n_obs=60, amplitude=0.30
                )
                
                features = classifier.extract_features(ztf)
                pred_type, confidence = classifier.classify(features)
                
                predictions[var_type] = (pred_type, confidence)
                self.log(f"  • {var_type}: {pred_type} (confidence: {confidence:.2f})")
            
            self.log("\n✓ Prediction tests PASSED", 'SUCCESS')
            return True
        
        except Exception as e:
            self.log(f"✗ Prediction tests FAILED: {e}", 'ERROR')
            return False
    
    def test_plotter(self) -> bool:
        """Test LightcurvePlotter component"""
        self.log("\n" + "=" * 80)
        self.log("TEST 5: LightcurvePlotter")
        self.log("=" * 80)
        
        try:
            self.log("\n  Generating test plots...", 'TEST')
            
            ztf = MockVariabilityGenerator.generate_ztf_lightcurve(
                'PLOT_TEST', 325.5, 65.9, 'irregular', n_obs=70, amplitude=0.35
            )
            
            analysis = VariabilityAnalysis(
                source_id='PLOT_TEST',
                source_type='NEOWISE',
                optical_detected=True,
                optical_n_obs=70,
                optical_mean_mag=18.5,
                optical_std_mag=0.30,
                optical_median_err=0.05,
                optical_snr=6.0,
                optical_trend=(0.001, 0.0005),
                infrared_detected=True,
                infrared_std_w1=0.25,
                infrared_std_w2=0.20,
                correlation=0.45,
                amplitude_ratio=0.83,
                timescale_agreement='good',
                variability_type='irregular',
                quality_flag='high'
            )
            
            plot_path = self.output_dir / 'test_single_lightcurve.png'
            LightcurvePlotter.plot_single_lightcurve(ztf, analysis, plot_path)
            self.log(f"  ✓ Single lightcurve plot: {plot_path}")
            
            neowise = MockVariabilityGenerator.generate_neowise_lightcurve(
                'PLOT_TEST', 325.5, 65.9
            )
            
            plot_path_mw = self.output_dir / 'test_multi_wavelength.png'
            LightcurvePlotter.plot_multi_wavelength(ztf, neowise, analysis, plot_path_mw)
            self.log(f"  ✓ Multi-wavelength plot: {plot_path_mw}")
            
            self.log("\n✓ Plotter tests PASSED", 'SUCCESS')
            return True
        
        except Exception as e:
            self.log(f"✗ Plotter tests FAILED: {e}", 'ERROR')
            return False
    
    def test_analysis_report(self) -> bool:
        """Test AnalysisReport component"""
        self.log("\n" + "=" * 80)
        self.log("TEST 6: AnalysisReport")
        self.log("=" * 80)
        
        try:
            self.log("\n  Generating analysis reports...", 'TEST')
            
            report = AnalysisReport(self.output_dir / 'reports')
            
            for i in range(10):
                var_type = ['stable', 'irregular', 'burst'][i % 3]
                
                ztf = MockVariabilityGenerator.generate_ztf_lightcurve(
                    f'REPORT_{i:02d}', 325.5 + i*0.1, 65.9,
                    variability_type=var_type, n_obs=50, amplitude=0.25
                )
                
                stats = VariabilityAnalyzer.calculate_statistics(ztf.mag, ztf.magerr)
                trend = VariabilityAnalyzer.detect_trend(ztf.mjd, ztf.mag, ztf.magerr)
                
                analysis = VariabilityAnalysis(
                    source_id=ztf.source_id,
                    source_type='NEOWISE',
                    optical_detected=True,
                    optical_n_obs=stats['n_observations'],
                    optical_mean_mag=stats['mean_magnitude'],
                    optical_std_mag=stats['std_magnitude'],
                    optical_median_err=stats['median_error'],
                    optical_snr=stats['snr'],
                    optical_trend=trend[:2] if trend else None,
                    infrared_detected=True,
                    infrared_std_w1=0.20,
                    infrared_std_w2=0.15,
                    correlation=np.random.uniform(0.2, 0.8),
                    amplitude_ratio=np.random.uniform(0.5, 1.5),
                    timescale_agreement='good' if np.random.random() > 0.3 else 'fair',
                    variability_type=var_type,
                    quality_flag='high' if stats['snr'] > 5 else 'medium' if stats['snr'] > 2 else 'low'
                )
                
                report.add_result(analysis)
            
            report.generate_summary_csv()
            report.generate_summary_statistics()
            report.generate_html_report()
            
            self.log(f"  ✓ Reports saved to {report.output_dir}")
            self.log("\n✓ AnalysisReport tests PASSED", 'SUCCESS')
            return True
        
        except Exception as e:
            self.log(f"✗ AnalysisReport tests FAILED: {e}", 'ERROR')
            return False
    
    def test_end_to_end_pipeline(self) -> bool:
        """Test complete end-to-end analysis pipeline"""
        self.log("\n" + "=" * 80)
        self.log("TEST 7: End-to-End Pipeline")
        self.log("=" * 80)
        
        try:
            self.log("\n  Running complete analysis pipeline on 30 sources...", 'TEST')
            
            ztf_lcs, neowise_lcs = MockVariabilityGenerator.generate_dataset(
                n_sources=30,
                variability_distribution={
                    'stable': 0.2,
                    'periodic': 0.2,
                    'irregular': 0.4,
                    'burst': 0.15,
                    'linear_trend': 0.05
                }
            )
            
            classifier = YSOVariabilityClassifier()
            analyzer = VariabilityAnalyzer()
            report = AnalysisReport(self.output_dir / 'pipeline_test')
            
            stats_summary = {
                'total': 0,
                'optical_detected': 0,
                'infrared_detected': 0,
                'variability_types': {},
                'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'snr_values': []
            }
            
            for ztf, neowise in zip(ztf_lcs, neowise_lcs):
                stats = VariabilityAnalyzer.calculate_statistics(ztf.mag, ztf.magerr)
                trend = VariabilityAnalyzer.detect_trend(ztf.mjd, ztf.mag, ztf.magerr)
                
                ir_stats_w1 = VariabilityAnalyzer.calculate_statistics(neowise.mag_w1, neowise.magerr_w1)
                ir_stats_w2 = VariabilityAnalyzer.calculate_statistics(neowise.mag_w2, neowise.magerr_w2)
                
                corr = LightcurveComparison.correlate_variability(
                    ztf.mag, neowise.mag_w1, ztf.mjd, neowise.mjd_w1
                )
                
                var_type = VariabilityAnalyzer.classify_variability_type(ztf.mag, ztf.mjd, trend)
                snr = stats['snr']
                quality = 'high' if snr > 5 else 'medium' if snr > 2 else 'low'
                
                analysis = VariabilityAnalysis(
                    source_id=ztf.source_id,
                    source_type='NEOWISE',
                    optical_detected=True,
                    optical_n_obs=stats['n_observations'],
                    optical_mean_mag=stats['mean_magnitude'],
                    optical_std_mag=stats['std_magnitude'],
                    optical_median_err=stats['median_error'],
                    optical_snr=snr,
                    optical_trend=trend[:2] if trend else None,
                    infrared_detected=True,
                    infrared_std_w1=ir_stats_w1['std_magnitude'],
                    infrared_std_w2=ir_stats_w2['std_magnitude'],
                    correlation=corr,
                    amplitude_ratio=ir_stats_w1['std_magnitude'] / stats['std_magnitude'] if stats['std_magnitude'] > 0 else 0,
                    timescale_agreement='good',
                    variability_type=var_type,
                    quality_flag=quality
                )
                
                report.add_result(analysis)
                
                stats_summary['total'] += 1
                stats_summary['optical_detected'] += 1
                stats_summary['infrared_detected'] += 1
                stats_summary['variability_types'][var_type] = stats_summary['variability_types'].get(var_type, 0) + 1
                stats_summary['quality_distribution'][quality] += 1
                stats_summary['snr_values'].append(snr)
            
            report.generate_summary_csv()
            report.generate_summary_statistics()
            report.generate_html_report()
            
            self.log(f"\n  Pipeline Results:")
            self.log(f"    • Total sources: {stats_summary['total']}")
            self.log(f"    • Optical detections: {stats_summary['optical_detected']}")
            self.log(f"    • Infrared detections: {stats_summary['infrared_detected']}")
            self.log(f"    • Mean SNR: {np.mean(stats_summary['snr_values']):.2f}")
            self.log(f"    • Variability types:")
            for vtype, count in stats_summary['variability_types'].items():
                self.log(f"      - {vtype}: {count}")
            self.log(f"    • Quality distribution:")
            for quality, count in stats_summary['quality_distribution'].items():
                self.log(f"      - {quality}: {count}")
            
            self.log("\n✓ End-to-End Pipeline tests PASSED", 'SUCCESS')
            return True
        
        except Exception as e:
            self.log(f"✗ End-to-End Pipeline tests FAILED: {e}", 'ERROR')
            import traceback
            self.log(traceback.format_exc(), 'ERROR')
            return False
    
    def run_all_tests(self) -> Dict:
        """Run all tests and generate summary"""
        
        self.log("\n" + "=" * 80)
        self.log("COMPLETE_ANALYSIS_FRAMEWORK TEST SUITE")
        self.log("=" * 80)
        
        test_results = {
            'VariabilityAnalyzer': self.test_variability_analyzer(),
            'LightcurveComparison': self.test_lightcurve_comparison(),
            'Classifier Features': self.test_classifier_features(),
            'Classifier Predictions': self.test_classifier_prediction(),
            'LightcurvePlotter': self.test_plotter(),
            'AnalysisReport': self.test_analysis_report(),
            'End-to-End Pipeline': self.test_end_to_end_pipeline()
        }
        
        self.log("\n" + "=" * 80)
        self.log("TEST SUMMARY")
        self.log("=" * 80)
        
        passed = sum(1 for v in test_results.values() if v)
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            self.log(f"  {test_name}: {status}")
        
        self.log(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            self.log("\n✓ ALL TESTS PASSED - FRAMEWORK IS OPERATIONAL", 'SUCCESS')
        else:
            self.log(f"\n⚠ {total - passed} test(s) failed", 'WARNING')
        
        log_file = self.output_dir / 'test_results.log'
        with open(log_file, 'w') as f:
            f.write('\n'.join(self.test_log))
        
        self.log(f"\nTest log saved to: {log_file}")
        
        return test_results


def main():
    """Run comprehensive test suite"""
    
    tester = FrameworkTester(output_dir='./framework_test_results')
    results = tester.run_all_tests()
    
    return results


if __name__ == '__main__':
    main()
