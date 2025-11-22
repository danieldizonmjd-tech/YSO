# NEOWISE - ZTF Optical Variability Analysis Report

## Executive Summary

This analysis examines the optical variability properties of Young Stellar Objects (YSOs) identified in two NEOWISE infrared survey papers, searching for corresponding detections in the Zwicky Transient Facility (ZTF) optical survey.

### Key Findings

**Paper 2: Neha & Sharma - NEOWISE Variability Study**
- **Total sources**: 20,654 YSOs with NEOWISE mid-infrared variability statistics
- **Variable sources (LCType ≠ NV)**: 5,444 sources (26.4%)
- **Non-variable sources**: 15,210 sources (73.6%)

**Paper 3: LAMOST - H-alpha Emission YSO Candidates**
- **Total sources**: 3,470 H-alpha emission YSO candidates
- **Geographic distribution**: RA 2.22° to 349.08°, Dec -9.94° to +79.14°

**Combined Catalog**
- **Total sources for ZTF analysis**: 24,124 sources
- **All have decimal RA/Dec coordinates** suitable for automated ZTF queries

---

## Paper 2 Analysis: Neha & Sharma

### Source Distribution by Light Curve Type

The NEOWISE variability catalog includes explicit light curve classifications (LCType):

| Type | Count | Percentage | Interpretation |
|------|-------|------------|-----------------|
| **NV** | 15,210 | 73.6% | Non-variable (steady in mid-IR) |
| **Irregular** | 4,103 | 19.9% | Erratic variability pattern |
| **Curved** | 586 | 2.8% | Smooth trend (brightening/fading) |
| **Burst** | 228 | 1.1% | Sudden outburst events |
| **Linear** | 215 | 1.0% | Linear brightness change |
| **Periodic** | 190 | 0.9% | Repeating variability |
| **Drop** | 122 | 0.6% | Sudden dimming events |

**Total Variable**: 5,444 sources (26.4%)

### YSO Evolutionary Classification

| Class | Count | Percentage |
|-------|-------|------------|
| ClassII | 12,757 | 61.8% |
| FS (F-star) | 4,070 | 19.7% |
| ClassI | 2,089 | 10.1% |
| ClassIII | 1,659 | 8.0% |
| Uncertain | 79 | 0.4% |

**Interpretation**: 
- ClassII sources are most common (disks around young stars)
- ClassI sources are younger, more embedded objects
- FS sources are F-type stars (potential early-type YSOs)
- ClassIII sources have dispersed disks (older YSOs)

---

## Paper 3 Analysis: LAMOST H-alpha Emission YSOs

### Source Profile

- **Total H-alpha emission YSO candidates**: 3,470
- **Discovery method**: Deep learning on LAMOST spectroscopy
- **Optical indicator**: H-alpha equivalent width (EW) measurements
- **All sources have precise RA/Dec coordinates**

These sources are strong candidates for optical variability because:
1. H-alpha emission indicates active accretion/jets (young systems)
2. Young systems commonly show optical variability
3. LAMOST provides systematic, unbiased YSO identification

---

## Data Quality for ZTF Queries

### Coordinate Precision

- **Paper 2**: RA/Dec in decimal degrees (9 decimal places)
- **Paper 3**: RA/Dec in decimal degrees (7-8 decimal places)
- **Precision**: Both allow matching radii of 1.5-2.5 arcseconds for ZTF point sources

### Optical Visibility Assessment

| Source Type | Paper | Count | ZTF Optical Detectability |
|-------------|-------|-------|--------------------------|
| Variable in mid-IR | Paper 2 | 5,444 | HIGH - already variable in 3-5 μm band |
| H-alpha emitting | Paper 3 | 3,470 | MODERATE-HIGH - active young systems |
| Non-variable mid-IR | Paper 2 | 15,210 | MODERATE - may still vary optically |

---

## Recommended ZTF Query Strategy

### Phase 1: High-Priority Targets

1. **Paper 2 Variable Sources**: 5,444 targets
   - Start with "Irregular" type sources (4,103 sources)
   - These show clear mid-IR variability
   - Likely to show correlated optical variability
   - Expected detection rate: 20-40% in ZTF

2. **LAMOST H-alpha Sources**: 3,470 targets
   - All are spectroscopically confirmed young systems
   - H-alpha indicates active accretion
   - Good candidates for optical time-series
   - Expected detection rate: 30-50% in ZTF

### Phase 2: Secondary Targets

3. **Burst/Outburst Sources**: 350 sources (Paper 2)
   - Known to have dramatic variability
   - Excellent for optical lightcurve comparison
   - Expected detection rate: 60-80%

4. **Periodic Sources**: 190 sources (Paper 2)
   - Possible rotators or eclipsing binaries
   - May show optical periodicity
   - Expected detection rate: 40-60%

### Phase 3: Control Sample

5. **Non-Variable Mid-IR Sources**: 15,210 sources
   - Control for background contamination
   - Assess intrinsic optical variability
   - Sample: every 10th source for statistical analysis

---

## Technical Approach for ZTF Queries

### Data Products Created

1. **ztf_query_master_list.csv** - All 24,124 sources with RA/Dec
2. **ztf_query_test_list.csv** - Subset (50 per paper) for initial testing
3. **paper2_variable_sources.csv** - 5,444 variable NEOWISE sources
4. **paper3_lamost_sources.csv** - 3,470 H-alpha emission sources

### Query Method

Use IRSA ZTF API with parameters:
```
POS=CIRCLE RA DEC 1.5_arcsec
BANDNAME=r (or g for comparison)
NOBS_MIN=10 (minimum 10 observations for lightcurve)
BAD_CATFLAGS_MASK=32768 (mask bad flags)
FORMAT=csv
```

### Expected Lightcurve Quality

- **Time baseline**: 2018 - present (~6 years)
- **Cadence**: 2-4 days typical for variable sources
- **Photometric precision**: 0.01-0.05 mag (r-band)
- **Typical measurements**: 20-200 per source

---

## Analysis Workflow

### Step 1: Coordinate Testing (✓ COMPLETED)

- ✓ Test coordinate provided: RA = 325.348403°, Dec = +65.927139°
- ✓ Parsed NEOWISE source catalogs
- ✓ Extracted coordinates from all three papers
- ✓ Created master source list

### Step 2: ZTF Query Execution (IN PROGRESS)

- Query first 50 sources from test list
- Download lightcurves in r-band
- Assess detection rate
- Evaluate photometric quality

### Step 3: Lightcurve Analysis

For detected sources:
1. Calculate variability statistics:
   - Standard deviation of magnitude
   - Median magnitude error
   - Signal-to-noise of variability

2. Identify variability types:
   - Irregular/random variability
   - Periodic/sinusoidal variation
   - Outburst/burst events
   - Secular trends

3. Compare optical ↔ infrared variability:
   - Timescale correlation
   - Amplitude ratio (optical vs mid-IR)
   - Phase relationships

### Step 4: Statistical Summary

- Variability occurrence rate by source type
- Optical vs infrared variability correlation
- Identification of exceptional systems
- Classification of variable types

---

## Expected Outcomes

### Optical Detection Rates

Based on YSO properties:
- **Paper 2 Variable sources**: 20-40% expected in ZTF
  - Irregular: 25-35%
  - Curved/Linear: 40-60%
  - Burst: 60-80%
  - Periodic: 50-70%

- **Paper 3 H-alpha sources**: 30-50% expected in ZTF
  - High probability due to young age

### Scientific Value

1. **Bridging Timescales**: Compare week-long infrared monitoring with nightly optical surveys
2. **Variability Mechanisms**: Identify accretion-driven vs rotation-driven variability
3. **Outburst Characterization**: Optical monitoring of infrared-selected outbursts
4. **Multi-wavelength Correlations**: Study how variability changes across spectrum

---

## Files Generated

### Data Files
- `paper2_all_sources.csv` - All 20,654 Neha & Sharma sources
- `paper2_variable_sources.csv` - 5,444 variable NEOWISE sources
- `paper3_lamost_sources.csv` - 3,470 LAMOST H-alpha sources
- `ztf_query_master_list.csv` - Combined 24,124-source catalog
- `ztf_query_test_list.csv` - 100-source test subset

### Analysis Scripts
- `extract_ztf_targets.py` - Source extraction and catalog creation
- `parse_neowise_sources.py` - Data parsing and validation
- `analyze_optical_visibility.py` - Optical property assessment
- `test_ztf_query.py` - ZTF API testing
- `analyze_ztf_neowise.py` - Full pipeline for ZTF queries and analysis

---

## Next Steps

1. **Resolve ZTF API Performance**: Use batch queries or alternative endpoints
2. **Run Lightcurve Queries**: Execute on priority source samples
3. **Generate Visualizations**: Create publication-quality lightcurve plots
4. **Statistical Analysis**: Quantify optical/infrared variability correlations
5. **Source Classification**: Identify specific variable star types

---

## References

**Paper 1**: Contreras Peña et al. - "Oh FUors where art thou" - Infrared FUor candidates

**Paper 2**: Neha & Sharma - "Illuminating Youth: Decades of Mid-Infrared Variability and Color Evolution of Young Stellar Objects"
- 20,654 YSOs with NEOWISE variability statistics
- Light curve classifications (Irregular, Curved, Burst, etc.)

**Paper 3**: Tan et al. - "A Catalog of Hα Emission Line Stars and 785 Newly Identified YSO Candidates from LAMOST"
- 3,470 H-alpha emission YSO candidates
- Deep learning classification on LAMOST spectra

---

**Report Generated**: 2025-11-21
**Analysis Status**: In Progress
**Next Update**: After ZTF query execution
