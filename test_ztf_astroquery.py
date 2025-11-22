#!/usr/bin/env python3
"""
Test ZTF query using astroquery
"""

import sys
import time

print("Testing astroquery availability...")
try:
    from astroquery.irsa import Irsa
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    print("✓ astroquery is available")
except ImportError as e:
    print(f"✗ astroquery not available: {e}")
    print("\nInstalling astroquery...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "astroquery", "-q"])
    from astroquery.irsa import Irsa
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    print("✓ astroquery installed")

def query_ztf_astroquery(ra, dec, radius=1.5):
    """
    Query ZTF using astroquery
    radius in arcseconds (default 1.5 arcsec)
    """
    print(f"\nQuerying ZTF for RA={ra}, Dec={dec}, radius={radius} arcsec")
    
    try:
        # Set up coordinates
        coord = SkyCoord(ra, dec, unit=(u.degree, u.degree))
        
        # Set up IRSA query for ZTF
        print("Sending query to IRSA ZTF database...")
        start = time.time()
        
        # Using a direct URL approach with astroquery
        table = Irsa.query_region(
            coord,
            radius=radius*u.arcsec,
            catalog='ZTF_DR11'  # Or try different catalog names
        )
        
        elapsed = time.time() - start
        print(f"✓ Query completed in {elapsed:.2f}s")
        print(f"Found {len(table)} sources")
        if len(table) > 0:
            print(table[:3])
        return table
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

print("=" * 80)
print("ZTF Query using astroquery")
print("=" * 80)

try:
    result = query_ztf_astroquery(325.348403, 65.927139)
    if result:
        print("\n✓ Query successful!")
    else:
        print("\n✗ Query failed or returned no results")
except Exception as e:
    print(f"\nError during query: {e}")
    import traceback
    traceback.print_exc()
