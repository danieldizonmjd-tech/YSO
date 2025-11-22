#!/usr/bin/env python3
"""
Mock Data Generator for Testing

Generates synthetic lightcurves matching real YSO variability characteristics
from NEOWISE paper classifications. Used to test the analysis framework
without requiring actual ZTF data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Dict
from COMPLETE_ANALYSIS_FRAMEWORK import ZTFLightcurve, NEOWISELightcurve


class MockVariabilityGenerator:
    """Generate synthetic lightcurves with realistic YSO properties"""
    
    @staticmethod
    def generate_ztf_lightcurve(source_id: str, ra: float, dec: float,
                               variability_type: str = 'irregular',
                               n_obs: int = 50, amplitude: float = 0.3) -> ZTFLightcurve:
        """
        Generate synthetic ZTF r-band lightcurve
        
        Args:
            source_id: Source identifier
            ra, dec: Coordinates
            variability_type: One of 'stable', 'periodic', 'irregular', 'burst', 'linear_trend'
            n_obs: Number of observations
            amplitude: Variability amplitude in magnitudes
        
        Returns:
            ZTFLightcurve object with synthetic data
        """
        
        mjd = np.sort(np.random.uniform(59000, 59500, n_obs))
        
        if variability_type == 'stable':
            mag = np.random.normal(18.5, 0.02, n_obs)
            
        elif variability_type == 'periodic':
            period = np.random.uniform(0.5, 10)
            phase = 2 * np.pi * (mjd - mjd[0]) / period
            mag = 18.5 + amplitude * np.sin(phase) + np.random.normal(0, 0.05, n_obs)
            
        elif variability_type == 'irregular':
            smooth = np.cumsum(np.random.normal(0, 0.02, n_obs))
            mag = 18.5 + amplitude * smooth / np.std(smooth) + np.random.normal(0, 0.05, n_obs)
            
        elif variability_type == 'burst':
            mag = np.random.normal(18.5, 0.05, n_obs)
            burst_idx = np.random.choice(n_obs, size=max(1, n_obs//10), replace=False)
            mag[burst_idx] -= amplitude * np.random.uniform(0.5, 1.0, len(burst_idx))
            
        elif variability_type == 'linear_trend':
            trend_slope = np.random.uniform(-0.01, 0.01)
            time_norm = (mjd - mjd[0]) / (mjd[-1] - mjd[0])
            mag = 18.5 + trend_slope * 500 * time_norm + np.random.normal(0, 0.05, n_obs)
            
        else:
            mag = np.random.normal(18.5, amplitude, n_obs)
        
        magerr = np.random.uniform(0.03, 0.08, n_obs)
        
        return ZTFLightcurve(
            source_id=source_id,
            ra=ra,
            dec=dec,
            mjd=mjd,
            mag=mag,
            magerr=magerr
        )
    
    @staticmethod
    def generate_neowise_lightcurve(source_id: str, ra: float, dec: float,
                                   amplitude_w1: float = 0.2,
                                   amplitude_w2: float = 0.15,
                                   n_obs: int = 30) -> NEOWISELightcurve:
        """
        Generate synthetic NEOWISE W1/W2 lightcurve
        
        Args:
            source_id: Source identifier
            ra, dec: Coordinates
            amplitude_w1, amplitude_w2: Variability amplitudes for W1, W2 bands
            n_obs: Number of observations per band
        
        Returns:
            NEOWISELightcurve object with synthetic data
        """
        
        mjd_w1 = np.sort(np.random.uniform(55000, 59500, n_obs))
        mjd_w2 = np.sort(np.random.uniform(55000, 59500, n_obs))
        
        smooth_w1 = np.cumsum(np.random.normal(0, 0.015, n_obs))
        smooth_w2 = np.cumsum(np.random.normal(0, 0.012, n_obs))
        
        mag_w1 = 12.5 + amplitude_w1 * smooth_w1 / np.std(smooth_w1) + np.random.normal(0, 0.03, n_obs)
        mag_w2 = 11.5 + amplitude_w2 * smooth_w2 / np.std(smooth_w2) + np.random.normal(0, 0.02, n_obs)
        
        magerr_w1 = np.random.uniform(0.02, 0.05, n_obs)
        magerr_w2 = np.random.uniform(0.02, 0.05, n_obs)
        
        return NEOWISELightcurve(
            source_id=source_id,
            ra=ra,
            dec=dec,
            mjd_w1=mjd_w1,
            mag_w1=mag_w1,
            magerr_w1=magerr_w1,
            mjd_w2=mjd_w2,
            mag_w2=mag_w2,
            magerr_w2=magerr_w2
        )
    
    @staticmethod
    def generate_dataset(n_sources: int = 100,
                        variability_distribution: Dict[str, float] = None) -> Tuple[List, List]:
        """
        Generate complete mock dataset
        
        Args:
            n_sources: Total number of sources
            variability_distribution: Dict mapping variability_type to fraction
                e.g., {'stable': 0.3, 'periodic': 0.2, 'irregular': 0.4, 'burst': 0.1}
        
        Returns:
            (ztf_lightcurves, neowise_lightcurves)
        """
        
        if variability_distribution is None:
            variability_distribution = {
                'stable': 0.25,
                'periodic': 0.25,
                'irregular': 0.35,
                'burst': 0.10,
                'linear_trend': 0.05
            }
        
        ztf_lcs = []
        neowise_lcs = []
        
        var_types = list(variability_distribution.keys())
        var_fractions = list(variability_distribution.values())
        var_counts = np.random.multinomial(n_sources, var_fractions)
        
        source_idx = 0
        for var_type, count in zip(var_types, var_counts):
            for _ in range(count):
                ra = np.random.uniform(0, 360)
                dec = np.random.uniform(-90, 90)
                source_id = f"YSO_MOCK_{source_idx:05d}"
                
                ztf_lc = MockVariabilityGenerator.generate_ztf_lightcurve(
                    source_id, ra, dec,
                    variability_type=var_type,
                    n_obs=np.random.randint(20, 100),
                    amplitude=np.random.uniform(0.05, 0.5)
                )
                ztf_lcs.append(ztf_lc)
                
                neowise_lc = MockVariabilityGenerator.generate_neowise_lightcurve(
                    source_id, ra, dec,
                    amplitude_w1=np.random.uniform(0.1, 0.4),
                    amplitude_w2=np.random.uniform(0.05, 0.3),
                    n_obs=np.random.randint(15, 50)
                )
                neowise_lcs.append(neowise_lc)
                
                source_idx += 1
        
        return ztf_lcs, neowise_lcs


class RealisticYSODataset:
    """Create mock dataset from real NEOWISE catalog statistics"""
    
    @staticmethod
    def create_from_paper2_distribution(csv_path: str, n_sample: int = 100) -> Tuple[pd.DataFrame, List, List]:
        """
        Load real sources from Paper 2 and assign mock lightcurves
        
        Args:
            csv_path: Path to paper2_variable_sources.csv
            n_sample: Number of sources to sample
        
        Returns:
            (catalog_df, ztf_lightcurves, neowise_lightcurves)
        """
        
        try:
            catalog = pd.read_csv(csv_path)
            if n_sample > 0:
                catalog = catalog.sample(n=min(n_sample, len(catalog)))
            
            ztf_lcs = []
            neowise_lcs = []
            
            for idx, row in catalog.iterrows():
                source_id = row.get('source_id', f'NEOWISE_{idx}')
                ra = row['ra']
                dec = row['dec']
                lc_type = row.get('variability_type', 'irregular')
                
                ztf_lc = MockVariabilityGenerator.generate_ztf_lightcurve(
                    source_id, ra, dec,
                    variability_type=lc_type,
                    n_obs=np.random.randint(30, 150),
                    amplitude=np.random.uniform(0.1, 0.4)
                )
                ztf_lcs.append(ztf_lc)
                
                neowise_lc = MockVariabilityGenerator.generate_neowise_lightcurve(
                    source_id, ra, dec,
                    n_obs=np.random.randint(20, 60)
                )
                neowise_lcs.append(neowise_lc)
            
            return catalog, ztf_lcs, neowise_lcs
        
        except FileNotFoundError:
            print(f"⚠ CSV file not found: {csv_path}")
            return None, [], []
    
    @staticmethod
    def create_test_subset(n_sources: int = 20) -> Tuple[List, List, pd.DataFrame]:
        """
        Create curated test dataset with known properties
        for validation
        
        Returns:
            (ztf_lightcurves, neowise_lightcurves, metadata_df)
        """
        
        metadata = {
            'source_id': [],
            'ra': [],
            'dec': [],
            'variability_type': [],
            'amplitude': [],
            'n_obs_ztf': [],
            'n_obs_neowise': []
        }
        
        ztf_lcs = []
        neowise_lcs = []
        
        var_types = ['stable', 'periodic', 'irregular', 'burst', 'linear_trend']
        
        for i in range(n_sources):
            var_type = var_types[i % len(var_types)]
            amplitude = np.random.uniform(0.05, 0.5)
            ra = 325.0 + np.random.uniform(-5, 5)
            dec = 65.0 + np.random.uniform(-5, 5)
            
            source_id = f"TEST_{var_type.upper()}_{i:02d}"
            
            ztf_lc = MockVariabilityGenerator.generate_ztf_lightcurve(
                source_id, ra, dec,
                variability_type=var_type,
                n_obs=50,
                amplitude=amplitude
            )
            ztf_lcs.append(ztf_lc)
            
            neowise_lc = MockVariabilityGenerator.generate_neowise_lightcurve(
                source_id, ra, dec,
                n_obs=40
            )
            neowise_lcs.append(neowise_lc)
            
            metadata['source_id'].append(source_id)
            metadata['ra'].append(ra)
            metadata['dec'].append(dec)
            metadata['variability_type'].append(var_type)
            metadata['amplitude'].append(amplitude)
            metadata['n_obs_ztf'].append(ztf_lc.n_observations)
            metadata['n_obs_neowise'].append(neowise_lc.n_observations_w1)
        
        return ztf_lcs, neowise_lcs, pd.DataFrame(metadata)


def quick_test():
    """Quick test of mock data generator"""
    
    print("=" * 70)
    print("MOCK DATA GENERATOR - QUICK TEST")
    print("=" * 70)
    
    print("\n1. Generating single lightcurves...")
    ztf = MockVariabilityGenerator.generate_ztf_lightcurve(
        'TEST_IRREGULAR', 325.5, 65.9, 'irregular', n_obs=50, amplitude=0.3
    )
    neowise = MockVariabilityGenerator.generate_neowise_lightcurve(
        'TEST_IRREGULAR', 325.5, 65.9
    )
    
    print(f"   ✓ ZTF lightcurve: {ztf.n_observations} observations")
    print(f"   ✓ NEOWISE lightcurve: {neowise.n_observations_w1} (W1) + {neowise.n_observations_w2} (W2) observations")
    
    print("\n2. Generating dataset of 100 sources...")
    ztf_lcs, neowise_lcs = MockVariabilityGenerator.generate_dataset(n_sources=100)
    print(f"   ✓ Created {len(ztf_lcs)} ZTF lightcurves")
    print(f"   ✓ Created {len(neowise_lcs)} NEOWISE lightcurves")
    
    print("\n3. Creating test subset...")
    ztf_test, neowise_test, metadata = RealisticYSODataset.create_test_subset(n_sources=20)
    print(f"   ✓ Test dataset: {len(ztf_test)} sources")
    print("\n   Variability types in test set:")
    for vtype, count in metadata['variability_type'].value_counts().items():
        print(f"     - {vtype}: {count}")
    
    print("\n" + "=" * 70)
    print("✓ Mock data generator ready for testing")
    print("=" * 70)


if __name__ == '__main__':
    quick_test()
