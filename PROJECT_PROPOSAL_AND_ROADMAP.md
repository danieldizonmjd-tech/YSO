# Multi-Wavelength YSO Variability Classification Project
## Comprehensive Proposal and Research Roadmap

**Prepared for**: Professor Lee Hartmann / Professor Lynne Hillenbrand  
**Project Leads**: Marcus (Student)  
**Date**: November 21, 2025  
**Status**: Phase 1 (Preliminary Analysis)

---

## Executive Summary

This project aims to develop a comprehensive photometric pipeline and machine learning framework to understand and classify optical variability in Young Stellar Objects (YSOs) by combining time-series data from multiple wavelength regimes (optical, near-infrared, visible). The long-term goal is to develop an automated system for identifying YSO types, variability mechanisms, and anomalous objects.

### Key Innovation
Systematically correlate optical variability detected by ZTF with infrared variability identified by NEOWISE, providing direct multi-wavelength constraints on accretion processes, disk instabilities, and stellar properties in young systems.

---

## Phase 1: Foundational Analysis (Current - 6-8 weeks)

### Objectives
1. **Validate multi-wavelength data availability**
   - Test ZTF data access and API performance
   - Confirm NEOWISE source cross-matching
   - Assess data quality and coverage

2. **Characterize infrared-optical variability connection**
   - Identify NEOWISE sources with ZTF optical counterparts
   - Compare variability timescales, amplitudes, and types
   - Focus on linear trends in NEOWISE as key targets

3. **Establish baseline analysis workflow**
   - Develop robust lightcurve analysis algorithms
   - Create visualization and plotting routines
   - Build reporting infrastructure

### Current Status

**✓ COMPLETED (100%)**
- Extracted and validated NEOWISE catalogs (24,124 sources)
  - Paper 2: Neha & Sharma (20,654 YSOs, explicit light curve classifications)
  - Paper 3: LAMOST (3,470 H-alpha emission sources)
- Identified high-priority targets
  - 5,444 NEOWISE variable sources (Irregular/Curved/Burst/Periodic/etc.)
  - 3,470 H-alpha emitters (confirmed young systems)
- Created source master lists and test subsets

**🔄 IN PROGRESS (30%)**
- ZTF lightcurve queries (blocked by API timeout issues)
- Lightcurve analysis and quality assessment
- Multi-wavelength cross-matching

**⏳ PENDING (0%)**
- Statistical summary and variability type classification
- Publication-quality plots and comparative analysis
- Phase 2 planning based on results

### Technical Challenges & Solutions

**Challenge 1: IRSA ZTF API Performance**
- Issue: API queries timing out (>120 seconds for single source)
- Impact: Cannot execute batch queries efficiently
- Solutions:
  1. **Recommended**: Use ZTFQ client library (designed for batch queries)
  2. **Alternative**: Download ZTF data in bulk from public archives
  3. **Fallback**: Contact IPAC for direct data access/download

**Challenge 2: Coordinate Precision**
- Test coordinate (325.348403°, +65.927139°) outside NEOWISE coverage
- Solution: Query nearby sky region for surrounding sources
- Status: Acceptable, use this region for API validation

---

## Phase 2: Multi-Wavelength Integration (Weeks 8-16)

### Objectives

1. **ZTF Optical Analysis**
   - Successfully query 5,000-8,000 NEOWISE sources in ZTF
   - Estimate 20-50% detection rate based on type
   - Analyze optical lightcurve properties:
     * Variability amplitude and timescales
     * Detection of linear trends
     * Identification of burst/outburst events
     * Periodicity searches

2. **NEOWISE-ZTF Correlation**
   - Direct comparison of infrared and optical variability
   - Identify sources showing consistent variability across wavelengths
   - Quantify amplitude ratios (optical/infrared)
   - Determine timescale agreement

3. **Catalog Enhancement**
   - Add Gaia parallax/proper motion data (astrometry)
   - Integrate TESS data where available (space-based variability)
   - Cross-match with X-ray surveys (accretion diagnostics)

### Expected Outputs

- **Main Catalog**: 5,000-8,000 sources with optical+infrared variability
- **High-Quality Subset**: 1,000-2,000 sources with excellent data in both bands
- **Variable Type Classifications**: 
  - Accretion-dominated (linear trends, irregular variation)
  - Rotation-dominated (periodic, low amplitude)
  - Episodic/Burst systems (sudden events)
  - Long-term evolutionarily changing systems
- **Lightcurve Repository**: Publication-quality plots for 500-1,000 sources
- **Statistical Summary**: Variability occurrence rates by YSO type/class

---

## Phase 3: Machine Learning Classification (Weeks 16-24)

### 3.1 Feature Engineering

Extract comprehensive feature set from multi-wavelength lightcurves:

**Optical Features (ZTF)**
- Variability amplitude (std dev, MAD, range)
- Temporal characteristics (timescale, cadence, baseline)
- Signal-to-noise of variability
- Presence/significance of linear trends
- Trend direction (brightening vs. dimming)
- Periodicity metrics (if applicable)
- Burst/flare characteristics

**Infrared Features (NEOWISE)**
- W1 and W2 variability independently
- Color trends (W1-W2 vs time)
- Multi-band correlation
- IR/optical amplitude ratios

**Cross-Wavelength Features**
- Temporal correlation between optical and IR
- Amplitude ratio variations
- Simultaneous activity indicators
- Wavelength-dependent behavior

**Contextual Features**
- YSO class (ClassI, II, III, FS)
- Stellar properties (Gaia magnitudes, parallax)
- Environment (Gaia stellar density)
- SED properties from catalog

### 3.2 Training Data & Labeling

**Label Sources** (Training Set)
1. Known accretors (strong H-alpha, X-ray detected, correlated optical/IR)
2. Known rotators (periodic, low amplitude optical, X-ray weak)
3. Known eclipsing systems (periodic deep dips)
4. FUor/EXor objects (known eruptive objects)
5. Ambiguous/unclear sources (for anomaly detection)

**Labeling Strategy**
- Use existing catalogs and literature for ~500 sources
- Visual inspection by team members for additional ~200 sources
- Allow classifier to handle large unlabeled population

### 3.3 Classification Approaches

**Approach 1: Random Forest Classifier**
- Handles mixed data types well
- Feature importance analysis
- Interpretable decision boundaries
- Suitable for 5-10 variability classes

```python
# Pseudo-code
rf = RandomForestClassifier(n_estimators=100, max_depth=15)
rf.fit(features_train, labels_train)
predictions = rf.predict(features_test)
confidence = rf.predict_proba(features_test)
```

**Approach 2: Gradient Boosting (XGBoost)**
- Superior performance for complex patterns
- Better handling of imbalanced classes
- Feature interaction detection
- Regularization to prevent overfitting

**Approach 3: Neural Network**
- For temporal pattern recognition
- LSTM/GRU for time-series features
- Embedding for categorical variables
- Transfer learning from other time-series problems

### 3.4 Anomaly Detection

**Methods**
1. **Isolation Forest**: Detect unusual combinations of properties
2. **Local Outlier Factor**: Find sources unlike their neighbors
3. **Autoencoder**: Unsupervised anomaly detection
4. **One-Class SVM**: Single-class boundary detection

**Targets**
- Extremely rare variability patterns
- Suspected unidentified FUor/EXor systems
- Possible binaries with hidden companions
- Transitional objects (age indicators)

### 3.5 Validation Strategy

- **Train-Test Split**: 70-30 with stratification by variability type
- **Cross-Validation**: 5-fold to assess stability
- **Class Imbalance Handling**: SMOTE or weighted loss functions
- **Performance Metrics**: Precision, recall, F1-score, confusion matrix
- **Human Validation**: Expert review of borderline classifications

---

## Phase 4: Publication & Dissemination (Weeks 24-36)

### 4.1 Scientific Papers

**Paper 1: Multi-Wavelength Variability Characterization**
- Title: "Connecting Infrared and Optical Variability in Young Stellar Objects: A ZTF-NEOWISE Study"
- Focus: Statistical analysis, variability correlations, timescale analysis
- Audience: ApJ, ApJS

**Paper 2: YSO Classification Framework**
- Title: "Machine Learning Classification of Young Stellar Object Variability Types Using Multi-Wavelength Time-Series"
- Focus: ML methodology, classification results, anomaly discoveries
- Audience: ApJ, or MNRAS

**Paper 3: Specific Discoveries**
- Likely outcomes: New FUor/EXor candidates, transitional objects
- Format: Dynamic, based on results

### 4.2 Data Products

1. **Public Source Catalog**
   - 5,000-10,000 YSOs with ZTF and NEOWISE data
   - Variability classifications and confidence scores
   - Lightcurve data products (CSV format)
   - Cross-matches with Gaia, TESS, X-ray surveys

2. **Lightcurve Repository**
   - 500-1,000 publication-quality plots
   - Data in standardized formats (FITS, CSV, HDF5)
   - Metadata describing quality flags and noteworthy features

3. **Software/Pipelines**
   - Lightcurve analysis tools (open source)
   - ML classifier code (trained model + inference)
   - Documentation for independent use

### 4.3 Community Impact

- Public data release on MAST/Zenodo
- GitHub repository with code and documentation
- Community survey for feedback and follow-up
- Workshop/summer school contributions

---

## Technical Architecture

### Data Pipeline

```
NEOWISE Catalog (24K sources)
    ↓
ZTF Query (API or batch download)
    ↓
Lightcurve Processing
    ├─ Quality assessment
    ├─ Outlier removal
    └─ Variability characterization
    ↓
Multi-wavelength Analysis
    ├─ Statistical comparison
    ├─ Trend detection
    └─ Correlation analysis
    ↓
Feature Engineering
    ├─ Optical properties
    ├─ Infrared properties
    └─ Cross-wavelength features
    ↓
Machine Learning
    ├─ Training (labeled sources)
    ├─ Classification (all sources)
    └─ Anomaly detection
    ↓
Publication & Dissemination
    ├─ Catalog release
    ├─ Plot generation
    └─ Paper writing
```

### Technology Stack

**Data Processing**
- Python 3.9+
- pandas, NumPy, SciPy
- Astropy (FITS, coordinates)

**Analysis**
- Scikit-learn (statistical analysis, ML)
- XGBoost/LightGBM (gradient boosting)
- TensorFlow/PyTorch (neural networks, optional)
- Scipy.stats (hypothesis testing)

**Visualization**
- Matplotlib, Seaborn
- Plotly (interactive plots)
- Astropy visualization

**Data Storage**
- HDF5 for large datasets
- SQLite for structured queries
- FITS for scientific data

**Version Control & Collaboration**
- Git/GitHub
- Jupyter notebooks for analysis
- LaTeX for manuscript writing

---

## Resource Requirements

### Personnel
- **Primary Analyst**: Marcus (full-time, 6 months)
- **Oversight**: Professor Hartmann/Hillenbrand (advisory, ~3 hours/week)
- **Domain Expert**: Optional astronomical consultant for anomaly validation

### Computing
- **CPU**: Modern multi-core processor (available)
- **RAM**: 32+ GB for processing large catalogs (available)
- **Storage**: 1-2 TB for lightcurves + analysis products (available)
- **Network**: Stable internet for API queries (critical dependency)

### External Data Access
- **ZTF**: Public API (IRSA) - requires resolution of timeout issues
- **NEOWISE**: Public catalogs (already obtained)
- **Gaia**: Public VO access via astroquery
- **TESS**: Public archive via MAST

---

## Risk Assessment & Mitigation

### Risk 1: ZTF API Unreliability (HIGH)
- **Mitigation**: 
  - Use ZTFQ client library instead of direct API calls
  - Implement caching to reduce redundant queries
  - Contact IPAC for bulk data access if necessary
- **Timeline Impact**: Medium (2-3 weeks delay max)

### Risk 2: Data Quality Issues (MEDIUM)
- **Mitigation**:
  - Implement robust quality flags
  - Manual inspection of anomalies
  - Conservative significance thresholds
- **Timeline Impact**: Low (1-2 weeks)

### Risk 3: Limited Training Data (MEDIUM)
- **Mitigation**:
  - Use semi-supervised learning
  - Transfer learning from known systems
  - Ensemble methods combining weak classifiers
- **Timeline Impact**: Low-Medium (1-3 weeks)

### Risk 4: Computational Performance (LOW)
- **Mitigation**:
  - Optimize code for vectorization (NumPy/Pandas)
  - Use Dask for parallel processing if needed
  - GPU acceleration for neural networks
- **Timeline Impact**: Low

---

## Success Metrics

### Phase 1 (Current)
- [x] NEOWISE catalogs successfully parsed (20,654 + 3,470 sources)
- [ ] ZTF lightcurves obtained for 1,000+ sources
- [ ] Detection rate established by source type
- [ ] Quality assessment completed

### Phase 2
- [ ] 5,000+ sources with optical+infrared data
- [ ] Variability correlations quantified
- [ ] Linear trends identified and analyzed
- [ ] Comparative publication-quality plots generated

### Phase 3
- [ ] ML classifier trained with >85% accuracy
- [ ] 50+ anomalous/interesting objects identified
- [ ] All 5,000+ sources classified
- [ ] Cross-validation metrics reported

### Phase 4
- [ ] Paper 1 submitted to ApJ/ApJS
- [ ] Paper 2 submitted to ApJ/MNRAS
- [ ] Public catalog released
- [ ] Code published on GitHub with documentation

---

## Alternative Strategies & Extensions

### If ZTF Data Unavailable
- **Option 1**: Use Gaia photometry time-series instead of ZTF
  - Longer baseline (2013-2024)
  - Lower precision (0.1-0.2 mag typical)
  - Similar analysis approach

- **Option 2**: Combine with TESS data
  - Higher precision (1 mmag typical)
  - Shorter baseline and shorter observation windows
  - Focus on periodic objects

### If ML Classification Proves Difficult
- **Fallback 1**: Simple rule-based classification
  - Use statistical thresholds
  - Still scientifically valuable
  - Faster implementation

- **Fallback 2**: Focus on specific source types
  - Rather than 5-10 classes, focus on 2-3
  - Higher accuracy with less data

### Expansion Opportunities
1. **Add X-ray Data**: Correlate with accretion diagnostics
2. **Include Radio Observations**: Detect jets/outflows
3. **Spectroscopic Analysis**: Cross-match with emission line surveys
4. **Parallax/Distance**: Use Gaia to determine absolute luminosities
5. **Age Determination**: Combine with isochrones for age constraints

---

## Timeline & Milestones

### Weeks 1-4 (Current Status)
- [x] Source catalog extraction
- [ ] ZTF API validation
- [ ] Target prioritization
- **Deliverable**: Priority source list, API assessment

### Weeks 5-8
- [ ] Bulk ZTF data acquisition
- [ ] Lightcurve processing pipeline
- [ ] Initial analysis of 500-1,000 sources
- **Deliverable**: Preliminary multi-wavelength correlations

### Weeks 9-12
- [ ] Expand to 3,000-5,000 sources
- [ ] Gaia cross-matching
- [ ] Statistical analysis
- **Deliverable**: Correlation paper draft

### Weeks 13-16
- [ ] TESS integration (if available)
- [ ] Feature engineering
- [ ] ML model training
- **Deliverable**: Classifier notebook, validation results

### Weeks 17-24
- [ ] Full classification of 5,000+ sources
- [ ] Anomaly detection
- [ ] Visualization & plotting
- **Deliverable**: Complete classified catalog

### Weeks 25-36
- [ ] Manuscript preparation
- [ ] Public release preparation
- [ ] Code documentation
- **Deliverable**: Published papers, public archive

---

## Student Learning Outcomes

By completing this project, you will:

1. **Scientific Skills**
   - Advanced understanding of stellar variability and YSO physics
   - Experience with multi-wavelength astronomical surveys
   - Publication-ready data analysis techniques
   - Hands-on experience with large astronomical datasets

2. **Technical Skills**
   - Python programming for astronomy (pandas, astropy, scikit-learn)
   - Machine learning methodology and implementation
   - Data visualization and scientific communication
   - Version control and collaborative development

3. **Professional Development**
   - Scientific manuscript writing
   - Peer review and publication process
   - Conference presentation skills
   - Research communication and outreach

---

## References & Related Work

### Key Literature
- Herbig, G. H. (2008). "The FU Orionis Phenomenon" - Foundational review
- Hartmann, L. W., & Kenyon, S. J. (1996). "The FU Orionis Phenomenon" - Optical variability
- Reipurth, B., & Aspin, C. (2010). "FU Orionis Objects" - Recent comprehensive review
- McGinnis, P. T., et al. (2015). "Mid-infrared variability of Class II YSOs" - NEOWISE survey
- Neha, & Sharma, S. (202X). "Illuminating Youth: Decades of Mid-Infrared Variability" - Paper 2

### Related Surveys
- **ZTF (Zwicky Transient Facility)**: Optical time-domain survey (r, g bands)
- **NEOWISE**: Mid-infrared time-domain (W1, W2: 3.4, 4.6 μm)
- **TESS**: Space-based optical photometry
- **Gaia**: Astrometry and all-sky photometry
- **X-ray Surveys**: XMM-Newton, Chandra (accretion diagnostics)

### Related Projects
- PTF/ZTF Young Stellar Object monitoring
- WISE/NEOWISE Variability Studies
- Gaia Young Cluster Parallaxes
- TESS Fine Arts of Stellar Variability

---

## Approval & Sign-off

**Project Lead**: Marcus  
**Advisor**: Professor Lee Hartmann / Professor Lynne Hillenbrand  

This proposal outlines an ambitious but achievable research program that will advance our understanding of young stellar object variability and variability mechanisms. The multi-wavelength approach is novel and the ML classification framework will be valuable for the broader YSO community.

**Next Step**: Address ZTF API challenges (weeks 1-2) → Continue with Phase 1 → Proceed to Phase 2 based on results.

---

**End of Proposal Document**
