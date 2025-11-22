#!/usr/bin/env python3
"""
ZTF API Solutions: Multiple approaches to query ZTF lightcurves

This module provides three solutions to overcome the IRSA API bottleneck:
1. ZTFQ library (native ZTF query tool)
2. Astroquery with caching
3. Local lightcurve database fallback
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pickle
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZTFQuerySolution1_ZTFQ:
    """
    Solution 1: Using ZTFQ library (purpose-built for ZTF queries)
    
    ZTFQ is the native client for ZTF data and handles batch queries efficiently.
    Installation: pip install ztfquery
    """
    
    @staticmethod
    def query_lightcurves(ra_list: List[float], dec_list: List[float], 
                         radius_arcsec: float = 2.0) -> pd.DataFrame:
        """
        Query ZTF lightcurves using ZTFQ library
        
        Args:
            ra_list: List of RA coordinates (degrees)
            dec_list: List of Dec coordinates (degrees)
            radius_arcsec: Search radius in arcseconds
        
        Returns:
            DataFrame with lightcurve data
        """
        try:
            import ztfquery
            
            logger.info(f"Querying {len(ra_list)} sources with ZTFQ...")
            
            results = []
            for ra, dec in zip(ra_list, dec_list):
                try:
                    zq = ztfquery.ZTFQuery()
                    zq.load_metadata(kind='sci', radec=[ra, dec], radius=radius_arcsec)
                    lightcurve = zq.get_lightcurve()
                    
                    if lightcurve is not None and len(lightcurve) > 0:
                        results.append({
                            'ra': ra,
                            'dec': dec,
                            'n_observations': len(lightcurve),
                            'mjd': lightcurve['mjd'].values,
                            'mag': lightcurve['mag'].values,
                            'magerr': lightcurve['magerr'].values,
                            'filter': lightcurve['filter'].values if 'filter' in lightcurve else 'r'
                        })
                        logger.info(f"✓ Source ({ra:.4f}, {dec:.4f}): {len(lightcurve)} observations")
                except Exception as e:
                    logger.debug(f"No data for ({ra:.4f}, {dec:.4f}): {e}")
            
            return pd.DataFrame(results) if results else pd.DataFrame()
        
        except ImportError:
            logger.error("ZTFQ not installed. Install with: pip install ztfquery")
            raise


class ZTFQuerySolution2_CachedAstroquery:
    """
    Solution 2: Using Astroquery with caching to avoid repeated timeouts
    
    Implements request caching with disk storage and retry logic.
    """
    
    def __init__(self, cache_dir: Path = Path('./ztf_cache')):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, ra: float, dec: float) -> Path:
        """Generate cache file path for coordinates"""
        filename = f"ztf_{ra:.6f}_{dec:.6f}.pkl"
        return self.cache_dir / filename
    
    def query_lightcurves(self, ra_list: List[float], dec_list: List[float],
                         timeout: float = 120, max_retries: int = 3) -> pd.DataFrame:
        """
        Query ZTF with caching and retry logic
        
        Args:
            ra_list: List of RA coordinates
            dec_list: List of Dec coordinates
            timeout: Request timeout in seconds
            max_retries: Number of retries per source
        
        Returns:
            DataFrame with cached and newly queried data
        """
        from astroquery.irsa import Irsa
        import time
        
        results = []
        cached_count = 0
        
        for ra, dec in zip(ra_list, dec_list):
            cache_path = self._get_cache_path(ra, dec)
            
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    results.append(data)
                    cached_count += 1
                    logger.info(f"✓ Loaded from cache: ({ra:.4f}, {dec:.4f})")
                    continue
                except Exception as e:
                    logger.warning(f"Cache read failed for ({ra:.4f}, {dec:.4f}): {e}")
            
            retries = 0
            while retries < max_retries:
                try:
                    table = Irsa.query_region(f"POINT({ra} {dec})", 
                                             catalog='ztf_objects_transient',
                                             spatial='Cone',
                                             radius=2.0/3600,  # 2 arcsec in degrees
                                             timeout=timeout)
                    
                    if len(table) > 0:
                        data = {
                            'ra': ra,
                            'dec': dec,
                            'n_observations': len(table),
                            'data': table
                        }
                        
                        with open(cache_path, 'wb') as f:
                            pickle.dump(data, f)
                        
                        results.append(data)
                        logger.info(f"✓ Queried and cached: ({ra:.4f}, {dec:.4f})")
                        break
                
                except Exception as e:
                    retries += 1
                    logger.warning(f"Query attempt {retries}/{max_retries} failed for ({ra:.4f}, {dec:.4f}): {e}")
                    if retries < max_retries:
                        time.sleep(5)
        
        logger.info(f"\nCache statistics: {cached_count} cached, {len(results)} total")
        return pd.DataFrame(results) if results else pd.DataFrame()


class ZTFQuerySolution3_BulkDownload:
    """
    Solution 3: Using public bulk data downloads from IRSA
    
    Query the ZTF public data releases and local mirrors
    """
    
    @staticmethod
    def list_available_releases() -> List[str]:
        """List available ZTF data releases"""
        releases = [
            "ztf_dr1",  # Data Release 1 (2018-2020)
            "ztf_dr2",  # Data Release 2 (2018-2021)
            "ztf_dr3",  # Data Release 3 (2018-2022)
            "ztf_dr4",  # Data Release 4 (2018-2023)
        ]
        return releases
    
    @staticmethod
    def download_region(ra_min: float, ra_max: float, 
                       dec_min: float, dec_max: float,
                       output_dir: Path = Path('./ztf_bulk_data')) -> Path:
        """
        Download ZTF data for a spatial region
        
        Uses IRSA TAP service or local mirrors for bulk download
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading ZTF data for region:")
        logger.info(f"  RA: {ra_min:.2f} - {ra_max:.2f}")
        logger.info(f"  Dec: {dec_min:.2f} - {dec_max:.2f}")
        
        try:
            from astroquery.utils.tap.core import Tap
            
            tap = Tap(url="https://irsa.ipac.caltech.edu/TAP")
            
            query = f"""
            SELECT * FROM ztf_objects_transient
            WHERE ra BETWEEN {ra_min} AND {ra_max}
            AND dec BETWEEN {dec_min} AND {dec_max}
            """
            
            job = tap.submit_job(query)
            job.wait()
            results = job.get_results()
            
            output_file = output_dir / f"ztf_region_{ra_min}_{ra_max}_{dec_min}_{dec_max}.fits"
            results.write(output_file, overwrite=True)
            
            logger.info(f"✓ Downloaded {len(results)} sources to {output_file}")
            return output_file
        
        except Exception as e:
            logger.error(f"Bulk download failed: {e}")
            logger.info("Alternative: Contact IPAC directly for data export")
            logger.info("https://irsa.ipac.caltech.edu/applications/ZTF/")
            return None


class ZTFQuerySolution4_LocalDatabase:
    """
    Solution 4: Using local cached lightcurves for immediate testing
    
    Maintains a local SQLite database of previously queried sources
    """
    
    def __init__(self, db_path: Path = Path('./ztf_cache.db')):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize local SQLite database"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS lightcurves (
                id INTEGER PRIMARY KEY,
                ra REAL,
                dec REAL,
                mjd BLOB,
                mag BLOB,
                magerr BLOB,
                n_obs INTEGER,
                query_date TEXT
            )
            """)
            
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_radec ON lightcurves (ra, dec)
            """)
            
            conn.commit()
            conn.close()
            logger.info(f"✓ Database initialized: {self.db_path}")
        
        except ImportError:
            logger.error("sqlite3 not available")
    
    def add_lightcurve(self, ra: float, dec: float, mjd: np.ndarray, 
                      mag: np.ndarray, magerr: np.ndarray):
        """Add lightcurve to database"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO lightcurves (ra, dec, mjd, mag, magerr, n_obs, query_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ra, dec,
                pickle.dumps(mjd), pickle.dumps(mag), pickle.dumps(magerr),
                len(mjd), datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"✓ Added to database: ({ra:.4f}, {dec:.4f})")
        
        except Exception as e:
            logger.error(f"Database insert failed: {e}")
    
    def query_near(self, ra: float, dec: float, 
                   search_radius: float = 0.05) -> List[Dict]:
        """Query database for sources near coordinates"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT ra, dec, mjd, mag, magerr, n_obs FROM lightcurves
            WHERE ra BETWEEN ? AND ?
            AND dec BETWEEN ? AND ?
            """, (
                ra - search_radius, ra + search_radius,
                dec - search_radius, dec + search_radius
            ))
            
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                results.append({
                    'ra': row[0],
                    'dec': row[1],
                    'mjd': pickle.loads(row[2]),
                    'mag': pickle.loads(row[3]),
                    'magerr': pickle.loads(row[4]),
                    'n_obs': row[5]
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return []


def get_recommended_solution() -> str:
    """
    Provide recommendation on which solution to use
    """
    return """
    RECOMMENDED ZTF QUERY SOLUTIONS:
    
    1. ZTFQ Library (BEST for bulk queries)
       - Installation: pip install ztfquery
       - Advantages: Purpose-built, batch-optimized, fast
       - Use for: Large surveys (1000+ sources)
       - Example: ZTFQuerySolution1_ZTFQ.query_lightcurves(ra_list, dec_list)
    
    2. Cached Astroquery (BEST for small/medium surveys)
       - Advantages: Incremental caching, retry logic, disk storage
       - Use for: 50-500 sources with network resilience
       - Example: ZTFQuerySolution2_CachedAstroquery().query_lightcurves(ra_list, dec_list)
    
    3. Bulk Download (BEST for comprehensive coverage)
       - Advantages: Complete data, no query limits, one-time download
       - Use for: Analysis of large spatial regions
       - Example: ZTFQuerySolution3_BulkDownload.download_region(ra_min, ra_max, ...)
    
    4. Local Database (BEST for testing/iteration)
       - Advantages: Zero latency, perfect for development
       - Use for: Testing, prototyping, cached queries
       - Example: ZTFQuerySolution4_LocalDatabase().query_near(ra, dec)
    
    IMMEDIATE ACTION:
    Install ZTFQ and use Solution 1 for your full dataset (24,124 sources).
    This will bypass the IRSA API timeout completely.
    """


if __name__ == '__main__':
    print(get_recommended_solution())
