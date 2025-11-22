#!/usr/bin/env python3
"""
Quick test of ZTF query with timeout handling
"""

import requests
import pandas as pd
from io import StringIO
import time

def test_ztf_query(ra, dec, timeout=10):
    """Test ZTF query with timeout"""
    url = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+{ra}+{dec}+0.000416667&BANDNAME=r&NOBS_MIN=10&BAD_CATFLAGS_MASK=32768&FORMAT=csv"
    
    print(f"\nTesting ZTF query for coordinates:")
    print(f"  RA = {ra}")
    print(f"  Dec = {dec}")
    print(f"  URL = {url}")
    
    try:
        start = time.time()
        print(f"\nSending request... (timeout={timeout}s)")
        response = requests.get(url, timeout=timeout)
        elapsed = time.time() - start
        
        print(f"Response received in {elapsed:.2f}s")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            # Try to parse CSV
            print(f"Response size: {len(response.text)} bytes")
            print(f"First 500 chars of response:\n{response.text[:500]}")
            
            try:
                df = pd.read_csv(StringIO(response.text))
                print(f"\n✓ Successfully parsed CSV with {len(df)} rows and {len(df.columns)} columns")
                print(f"Columns: {df.columns.tolist()}")
                if len(df) > 1:
                    print(f"First few rows:\n{df.head()}")
                return True
            except Exception as e:
                print(f"✗ Error parsing CSV: {e}")
                return False
        else:
            print(f"✗ HTTP Error {response.status_code}")
            return False
            
    except requests.Timeout:
        print(f"✗ Request timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == '__main__':
    print("=" * 80)
    print("ZTF Query Test")
    print("=" * 80)
    
    # Test with provided coordinates
    success = test_ztf_query(325.348403, 65.927139, timeout=30)
    
    if success:
        print("\n✓ ZTF query is working!")
    else:
        print("\n✗ ZTF query failed. This may indicate a network issue or API problem.")
    
    print("=" * 80)
