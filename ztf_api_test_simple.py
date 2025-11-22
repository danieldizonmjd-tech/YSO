#!/usr/bin/env python3
"""
Simple test to diagnose ZTF API issue
"""

import requests
from io import StringIO
import pandas as pd
import time

print("Testing ZTF API connectivity...")
print("=" * 80)

# Test 1: Simple HEAD request to check if server is responding
print("\n1. Testing server connectivity (HEAD request)...")
try:
    response = requests.head("https://irsa.ipac.caltech.edu/", timeout=10)
    print(f"   Status: {response.status_code}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Try with a very short timeout first
print("\n2. Testing with short timeout (5s)...")
ra, dec = 194.3, -63.0  # From catalog (guaranteed to exist)
url = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+{ra}+{dec}+0.000416667&BANDNAME=r&NOBS_MIN=10&BAD_CATFLAGS_MASK=32768&FORMAT=csv"

try:
    print(f"   URL: {url[:100]}...")
    response = requests.get(url, timeout=5)
    print(f"   Response status: {response.status_code}")
except requests.Timeout:
    print(f"   ✗ Timeout after 5s")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Try with longer timeout and measure
print("\n3. Testing with longer timeout (30s)...")
start = time.time()
try:
    print(f"   Sending request...")
    response = requests.get(url, timeout=30)
    elapsed = time.time() - start
    print(f"   Response received in {elapsed:.1f}s")
    print(f"   Status: {response.status_code}")
    print(f"   Content length: {len(response.text)} bytes")
    if len(response.text) < 500:
        print(f"   Content preview: {response.text[:200]}")
except requests.Timeout:
    elapsed = time.time() - start
    print(f"   ✗ Timeout after {elapsed:.1f}s")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 80)
print("Assessment:")
print("  - If timeout occurs, IRSA API is too slow for batch queries")
print("  - Alternative: Use ZTFQ client library or cached ZTF data")
print("  - Workaround: Implement manual lightcurve downloading with longer waits")
