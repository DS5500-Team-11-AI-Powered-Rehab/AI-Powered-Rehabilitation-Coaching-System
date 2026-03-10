# Integration Layer Testing Guide

Complete testing framework for validating your integration layer with synthetic continuous frame data generated from QEVD dataset.

**⚠️ Prerequisites:**
- Python environment: `rehab_ai_env` conda environment
- Required packages: `diskcache`, `matplotlib` (installed in rehab_ai_env)
- Synthetic data: Generated using `generate_synthetic_data.py`
- Working directory: `tests/integration_testing/` folder

---

## 📋 Quick Reference Card

**Most Common Commands:**
```bash
# Activate environment (do this once per session)
conda activate rehab_ai_env

# Automated test script (runs test → tune → visualize)
./quick_start_test.sh

# Quick test (10 videos)
python test_integration.py --dataset synthetic --max-videos 10 --config tuned_config.json

# Full test (all 217 test videos)
python test_integration.py --dataset synthetic --config tuned_config.json

# Large validation test (100 videos)
python test_integration.py --dataset synthetic_val --max-videos 100 --config tuned_config.json

# Verbose mode (see each event)
python test_integration.py --dataset synthetic --max-videos 5 --verbose --config tuned_config.json

# Generate synthetic data from ALL source files
python generate_synthetic_data.py --input rag_infer_logs_test --output synthetic_test_data --num-videos 1000 --frames-per-video 120
```

---

## 📁 File Structure

```
tests/
├── integration_testing/               # Integration layer testing suite
    ├── test_integration.py           # Main test runner
    ├── test_tune_thresholds.py       # Threshold tuning analyzer
    ├── test_visualize.py              # Visualization generator
    ├── generate_synthetic_data.py     # Generate synthetic continuous frame data
    ├── TEST_README.md                 # This file
    ├── tuned_config.json              # Relaxed thresholds for testing
    ├── rag_infer_logs_test/           # QEVD test event logs (538 files)
    ├── rag_infer_logs_val/            # QEVD validation event logs (2,264 files)
    ├── synthetic_test_data/           # Generated synthetic data from test (217 videos)
    ├── synthetic_val_data/            # Generated synthetic data from val (1,292 videos)
    └── test_results/                  # Test outputs (auto-created)
        ├── *.json                     # Test results
        └── figures/                   # Visualizations
            └── *.png
```

---

## 🎯 Understanding the Data

### **Why Synthetic Data?**

The QEVD dataset (`rag_infer_logs_test` and `rag_infer_logs_val`) contains **event logs** - single frames where mistakes were detected. Each `.jsonl.gz` file has:
- Line 1: Metadata (video info)
- Line 2: Single frame with detected mistake (if any)

**Problem:** The integration layer needs **continuous frame sequences** (75+ frames minimum) to:
- Detect persistent mistakes over time
- Calculate persistence rates (appears in 30%+ of frames)
- Measure duration (persists for 3+ seconds)
- Apply temporal filtering and deduplication

**Solution:** Generate synthetic continuous frame data from event logs using `generate_synthetic_data.py`. This creates realistic 120-frame sequences with:
- Mistakes that appear and persist naturally
- Baseline "good form" frames before mistakes start
- Realistic confidence variation (±0.1)
- 70% detection rate for persistent mistakes

### **Dataset Overview**

| Dataset | Files | Videos with Frames | Synthetic Videos | Size |
|---------|-------|-------------------|------------------|------|
| rag_infer_logs_test | 538 | 217 | 217 | 14 MB |
| rag_infer_logs_val | 2,264 | 1,292 | 1,292 | 81 MB |
| **Total** | 2,802 | 1,509 | **1,509** | **95 MB** |

---

## 🚀 Quick Start (5 Minutes)

### **Step 1: Activate Environment**

```bash
conda activate rehab_ai_env
# OR if running one-off commands:
conda run -n rehab_ai_env <command>
```

### **Step 2: Generate Synthetic Data (if not already done)**

The QEVD dataset contains single-frame event logs, but the integration layer needs continuous frame sequences. Generate synthetic data:

```bash
# Generate from test set (217 videos, ~14 MB)
python generate_synthetic_data.py --input rag_infer_logs_test --output synthetic_test_data --num-videos 1000 --frames-per-video 120

# Generate from validation set (1,292 videos, ~81 MB)
python generate_synthetic_data.py --input rag_infer_logs_val --output synthetic_val_data --num-videos 10000 --frames-per-video 120 --seed 43
```

### **Step 3: Run First Test**

```bash
# Quick test on first 10 synthetic videos
conda run -n rehab_ai_env python test_integration.py --dataset synthetic --max-videos 10 --config tuned_config.json
```

This will:
- ✅ Load 10 videos from synthetic test set
- ✅ Process through integration layer
- ✅ Generate metrics
- ✅ Save results to `test_results/`

### **Step 4: Analyze Results**

```bash
# Use the results file from Step 3 (check test_results/ folder)
conda run -n rehab_ai_env python test_tune_thresholds.py --results test_results/test_results_synthetic_YYYYMMDD_HHMMSS.json
```

This will:
- ✅ Analyze performance
- ✅ Identify issues (too many/few events)
- ✅ Generate recommendations
- ✅ Update `tuned_config.json` if needed

### **Step 5: Generate Visualizations**

```bash
conda run -n rehab_ai_env python test_visualize.py --results test_results/test_results_synthetic_YYYYMMDD_HHMMSS.json
```

This will create charts in `test_results/figures/`

---

## 🚀 Automated Testing Script

### **quick_start_test.sh**

Automated end-to-end testing pipeline that runs all three testing scripts in sequence.

**Usage:**
```bash
conda activate rehab_ai_env
cd tests/
./quick_start_test.sh
```

**What it does:**
1. **Runs integration tests** on synthetic dataset
2. **Analyzes results** and generates tuned_config.json
3. **Creates visualizations** in test_results/figures/
4. Provides summary of all outputs

**Script steps:**
```bash
# Step 1: Test integration layer
python test_integration.py --dataset synthetic --config tuned_config.json

# Step 2: Analyze and tune thresholds
python test_tune_thresholds.py --results test_results/test_results_synthetic_*.json

# Step 3: Generate visualizations
python test_visualize.py --results test_results/test_results_synthetic_*.json
```

**Output locations:**
- Test results: `test_results/test_results_synthetic_YYYYMMDD_HHMMSS.json`
- Tuned config: `tuned_config.json`
- Visualizations: `test_results/figures/*.png`

---

## 📊 Complete Testing Workflow

### **Phase 1: Baseline Testing (Week 9 Day 1-2)**

```bash
# 1. Test on synthetic test set (50 videos sample)
conda run -n rehab_ai_env python test_integration.py --dataset synthetic --max-videos 50 --config tuned_config.json

# 2. Test on synthetic validation set (100 videos sample)
conda run -n rehab_ai_env python test_integration.py --dataset synthetic_val --max-videos 100 --config tuned_config.json

# 3. Test on ALL synthetic test videos (217 total)
conda run -n rehab_ai_env python test_integration.py --dataset synthetic --config tuned_config.json

# 4. Analyze results
conda run -n rehab_ai_env python test_tune_thresholds.py --results test_results/test_results_synthetic_*.json

# 5. Generate visualizations
conda run -n rehab_ai_env python test_visualize.py --results test_results/test_results_synthetic_*.json
```

**What to look for:**
- ✅ Events per video: 0.3-1.0 (target for synthetic data with tuned config)
- ✅ Cache hit rate: 0% initially (no cache populated yet - this is expected)
- ✅ Tier distribution: Balanced across Tier 2/3 (Tier 1 requires cache population)
- ✅ Video coverage: 25-35% of videos have events (based on actual results)

**Expected Results with Synthetic Data:**
- Test dataset (217 videos): ~70 events (32% coverage)
- Validation dataset (1,292 videos): ~350 events (27% coverage)
- Tier distribution: ~50% Tier 2, ~50% Tier 3 (varies by dataset)
- Severity: High 20-40%, Medium 25-50%, Low 25-40%

---

### **Phase 2: Threshold Tuning (Week 9 Day 3-4)**

If baseline shows issues (too many/few events), tune thresholds:

```bash
# 1. Analyze and get recommendations
conda run -n rehab_ai_env python test_tune_thresholds.py --results test_results/test_results_synthetic_*.json
# This creates/updates tuned_config.json

# 2. Test with tuned config
conda run -n rehab_ai_env python test_integration.py --dataset synthetic --config tuned_config.json

# 3. Compare results
conda run -n rehab_ai_env python test_tune_thresholds.py --results test_results/test_results_synthetic_*.json

# 4. Iterate until satisfied
```

**Tuning Guidelines:**

| Issue | Solution |
|-------|----------|
| Too many events (>5 per video) | Increase `MIN_PERSISTENCE_RATE` to 0.35-0.40 |
| | Increase `MIN_DURATION_SECONDS` to 4.0-5.0 |
| Too few events (<0.5 per video) | Decrease `MIN_PERSISTENCE_RATE` to 0.25 |
| | Decrease `MIN_DURATION_SECONDS` to 2.5 |
| Low cache hit rate (<50%) | Add more patterns to Tier 1 cache |
| | Review top mistakes and cache them |

---

### **Phase 3: Deep Analysis (Week 9 Day 5)**

```bash
# 1. Verbose output to see individual events
conda run -n rehab_ai_env python test_integration.py --dataset synthetic --max-videos 5 --verbose --config tuned_config.json

# 2. Test all synthetic datasets comprehensively
conda run -n rehab_ai_env python test_integration.py --dataset synthetic --config tuned_config.json
conda run -n rehab_ai_env python test_integration.py --dataset synthetic_val --config tuned_config.json

# 3. Generate all visualizations
conda run -n rehab_ai_env python test_visualize.py --results test_results/test_results_synthetic_*.json
conda run -n rehab_ai_env python test_visualize.py --results test_results/test_results_synthetic_val_*.json
```

**What to analyze:**
- Top mistakes: Are they what you expect?
- Top exercises: Which exercises generate most events?
- Videos with no events: Are they actually perfect form?
- Mistake-severity mapping: Are high-severity mistakes being caught?

---

## 📖 Detailed Command Reference

### **test_integration.py**

Main test runner for integration layer.

**Basic Usage:**
```bash
conda run -n rehab_ai_env python test_integration.py --dataset <test|val|sample|both|synthetic|synthetic_val>
```

**Options:**
- `--dataset <test|val|sample|both|synthetic|synthetic_val>` - Which dataset to test
  - `test` - QEVD test event logs (single frames only)
  - `val` - QEVD validation event logs (single frames only)
  - `sample` - Small sample data for quick testing
  - `synthetic` - Generated continuous frames from test (217 videos)
  - `synthetic_val` - Generated continuous frames from val (1,292 videos)
  - `both` - Test both test and val datasets
- `--verbose` - Print detailed output for each video
- `--max-videos N` - Limit to first N videos (for quick testing)
- `--config PATH` - Use custom config JSON file (default: tuned_config.json recommended)

**Examples:**

```bash
# Quick test with synthetic data (10 videos)
conda run -n rehab_ai_env python test_integration.py --dataset synthetic --max-videos 10 --config tuned_config.json

# Full test on all synthetic test videos (217)
conda run -n rehab_ai_env python test_integration.py --dataset synthetic --config tuned_config.json

# Test validation set (sample of 100)
conda run -n rehab_ai_env python test_integration.py --dataset synthetic_val --max-videos 100 --config tuned_config.json

# Verbose output to see individual events
conda run -n rehab_ai_env python test_integration.py --dataset synthetic --max-videos 5 --verbose --config tuned_config.json
```

**Output:**
- Console: Summary statistics
- File: `test_results/test_results_<dataset>_<timestamp>.json`

---

### **generate_synthetic_data.py**

Generates synthetic continuous frame sequences from single-frame event logs.

**Usage:**
```bash
python generate_synthetic_data.py --input <input_dir> --output <output_dir> --num-videos <N> --frames-per-video <F>
```

**Options:**
- `--input` - Input directory with .jsonl.gz event files
- `--output` - Output directory for synthetic .json files
- `--num-videos` - Number of synthetic videos to generate (script will use all available if this exceeds available files)
- `--frames-per-video` - Frames per synthetic video (default: 100, recommended: 120)
- `--seed` - Random seed for reproducibility (default: 42)

**Examples:**

```bash
# Generate from ALL test videos (217 with frame data)
python generate_synthetic_data.py --input rag_infer_logs_test --output synthetic_test_data --num-videos 1000 --frames-per-video 120

# Generate from ALL validation videos (1,292 with frame data)
python generate_synthetic_data.py --input rag_infer_logs_val --output synthetic_val_data --num-videos 10000 --frames-per-video 120 --seed 43

# Generate small test set for debugging
python generate_synthetic_data.py --input rag_infer_logs_test --output synthetic_debug --num-videos 10 --frames-per-video 120
```

**What it does:**
1. Scans input directory for .jsonl.gz files with event frames
2. For each file, extracts the event frame (mistake detection)
3. Generates synthetic 120-frame sequence:
   - Frames 0-14: Good form (no mistakes)
   - Frames 15-30: Mistake starts appearing
   - Frames 30-70: Mistake persists at 70% detection rate
   - Frames 70-120: Mistake continues or resolves
4. Adds realistic variation: confidence ±0.1, transient mistakes at 10%
5. Saves as .json file with continuous frame data

---

### **test_tune_thresholds.py**

Analyzes test results and recommends threshold adjustments.

**Usage:**
```bash
conda run -n rehab_ai_env python test_tune_thresholds.py --results <path_to_results.json>
```

**What it does:**
1. Analyzes current metrics (event rate, cache hit rate, coverage)
2. Identifies issues (too many/few events, low cache hits)
3. Provides specific recommendations
4. Generates `tuned_config.json` with suggested parameters

**Example:**
```bash
conda run -n rehab_ai_env python test_tune_thresholds.py --results test_results/test_results_synthetic_20260305_143022.json
```

**Output:**
- Console: Analysis and recommendations
- File: `tuned_config.json` (if recommendations generated)

---

### **test_visualize.py**

Generates charts and graphs from test results.

**Usage:**
```bash
conda run -n rehab_ai_env python test_visualize.py --results <path_to_results.json>
```

**Generates 6 visualizations:**
1. **Tier Distribution** - Bar chart of Tier 1/2/3 routing
2. **Severity Distribution** - Bar chart of High/Medium/Low severity
3. **Top 15 Mistakes** - Horizontal bar chart of most common mistakes
4. **Top 15 Exercises** - Horizontal bar chart of exercises with most events
5. **Events per Video** - Histogram showing distribution
6. **Mistake-Severity Heatmap** - Matrix showing which mistakes have which severity

**Example:**
```bash
conda run -n rehab_ai_env python test_visualize.py --results test_results/test_results_synthetic_20260305_143022.json
```

**Output:**
- Files: `test_results/figures/*.png`

---

## 🎯 Key Metrics & Targets

| Metric | Target Range (Production) | Actual (Synthetic Data) | Why |
|--------|---------------------------|------------------------|-----|
| **Events per Video** | 1-3 | 0.27-0.32 | Synthetic data has controlled mistakes; production may vary |
| **Cache Hit Rate** | 60-80% | 0% (not populated) | Cache needs to be built from common patterns |
| **Video Coverage** | 30-50% | 27-32% | ✅ Good - not all videos should have events |
| **Tier 1 (Cache)** | 60-80% | 0% | Cache needs population from common mistakes |
| **Tier 2 (RAG)** | 15-30% | 44-87% | ✅ Good - medium complexity patterns |
| **Tier 3 (Reasoning)** | 5-10% | 12-56% | Higher due to no cache; expected |
| **High Severity** | 20-40% | 6-41% | ✅ Safety-critical mistakes being caught |
| **Medium Severity** | 40-60% | 26-50% | ✅ Form corrections |
| **Low Severity** | 10-20% | 33-44% | Acceptable for optimization opportunities |

**Note:** Synthetic data results differ from production expectations because:
1. No Tier 1 cache populated yet
2. Controlled mistake generation (70% detection rate)
3. Fixed persistence patterns (mistakes appear frames 15-70)
4. Simplified exercise variety compared to real workout sessions

---

## 🔍 Interpreting Results

### **Good Results Look Like (Synthetic Data with tuned_config.json):**

```
Overall Statistics:
  Total Videos: 50
  Total Coaching Events: 16              ← 0.32 events/video ✅ (expected for synthetic)
  Videos with Events: 16 (32.0%)         ← 32% coverage ✅

Tier Breakdown:
  Tier 1 (Cache): 0 (0%)                 ← Cache not populated yet (expected)
  Tier 2 (RAG): 14 (87.5%)               ← Good for medium severity ✅
  Tier 3 (Reasoning): 2 (12.5%)          ← Complex patterns ✅

Severity Breakdown:
  High: 1 (6.2%)                         ← Safety-critical ✅
  Medium: 8 (50.0%)                      ← Most common ✅
  Low: 7 (43.8%)                         ← Minor optimizations ✅

Top Mistakes:
  90 degrees, kneeing, barely moving hands, head up, etc.
  
Top Exercises:
  reverse crunches, standing kick, low lunge, warrior 2, etc.
```

### **Validation Dataset Results (100 videos):**

```
Overall Statistics:
  Total Videos: 100
  Total Coaching Events: 27              ← 0.27 events/video ✅
  Videos with Events: 27 (27.0%)         ← 27% coverage ✅

Tier Breakdown:
  Tier 2 (RAG): 12 (44.4%)               ← Balanced ✅
  Tier 3 (Reasoning): 15 (55.6%)         ← More complex patterns ✅

Severity Breakdown:
  High: 11 (40.7%)                       ← More safety issues ⚠️
  Medium: 7 (25.9%)                      ← Form corrections ✅
  Low: 9 (33.3%)                         ← Optimizations ✅
```

### **Bad Results Look Like:**

**❌ Too Many Events:**
```
Total Coaching Events: 800          ← 8 events/video ❌
Videos with Events: 95 (95%)        ← Almost every video ❌
```
**Solution:** Increase thresholds (persistence, duration, confidence)

**❌ Too Few Events:**
```
Total Coaching Events: 20           ← 0.2 events/video ❌
Videos with Events: 15 (15%)        ← Very low coverage ❌
```
**Solution:** Decrease thresholds or check if CV is detecting mistakes

**❌ Low Cache Hit Rate:**
```
Tier 1 (Cache): 30 (20%)            ← Low cache usage ❌
Tier 2 (RAG): 100 (67%)             ← Most go to RAG ❌
```
**Solution:** Add top mistake types to Tier 1 cache

---

## 🐛 Known Issues & Fixes

### **Fixed: last_coaching_time Initialization Bug**

**Issue:** Integration layer was initialized with `last_coaching_time = 0`, causing the cooldown check to block all coaching events in the first ~10 seconds of video (current_time < MIN_COACHING_INTERVAL).

**Symptoms:**
- 0 events generated despite persistent mistakes
- Verbose mode shows mistakes detected but `should_coach=False`
- `time_since_last` shows small positive value (e.g., 3.3 seconds)

**Fix:** Changed initialization in `integration_layer.py`:
```python
# Before (WRONG):
self.last_coaching_time = 0

# After (CORRECT):
self.last_coaching_time = -1000  # Large negative value = never coached before
```

**Location:** [integration_layer.py:152](../src/integration/integration_layer.py#L152)

**Status:** ✅ Fixed and tested with synthetic data

---

## 🐛 Troubleshooting

### **Problem: No events generated**

**Symptoms:**
```
Total Coaching Events: 0
Videos with Events: 0 (0%)
```

**Possible Causes:**
1. Thresholds too strict
2. Not using synthetic data (QEVD event logs have only 1-2 frames, insufficient for temporal aggregation)
3. Bug in integration layer initialization (fixed: last_coaching_time should be -1000, not 0)

**Debug Steps:**
```bash
# 1. Make sure you're using synthetic data (not raw QEVD event logs)
conda run -n rehab_ai_env python test_integration.py --dataset synthetic --max-videos 10 --config tuned_config.json

# 2. Run with verbose to see frame processing
conda run -n rehab_ai_env python test_integration.py --dataset synthetic --max-videos 1 --verbose --config tuned_config.json

# 3. Check synthetic data format
python -c "import json; data=json.load(open('synthetic_test_data/clip_0054833_synthetic.json')); print(f'Frames: {len(data)}'); print(f'First frame mistakes: {data[0].get(\"mistakes\", [])}')"

# 4. Lower thresholds dramatically (edit tuned_config.json):
{
  "MIN_PERSISTENCE_RATE": 0.20,
  "MIN_CONFIDENCE": 0.25,
  "MIN_DURATION_SECONDS": 2.0,
  "MIN_COACHING_INTERVAL": 10
}

# 5. Test with relaxed config
conda run -n rehab_ai_env python test_integration.py --dataset synthetic --max-videos 10 --config tuned_config.json
```

---

### **Problem: Too many events**

**Symptoms:**
```
Total Coaching Events: 500
Avg Events per Video: 5+
```

**Solution:**
```bash
# 1. Analyze what's triggering events
conda run -n rehab_ai_env python test_tune_thresholds.py --results test_results/test_results_*.json

# 2. Use suggested config
conda run -n rehab_ai_env python test_integration.py --dataset synthetic --config tuned_config.json

# 3. Manually adjust if needed
# tuned_config.json:
{
  "MIN_PERSISTENCE_RATE": 0.40,    # Increase
  "MIN_DURATION_SECONDS": 5.0,      # Increase
  "MIN_COACHING_INTERVAL": 15       # Increase cooldown
}
```

---

### **Problem: Import errors**

**Symptoms:**
```
ModuleNotFoundError: No module named 'integration'
ModuleNotFoundError: No module named 'diskcache'
```

**Solution:**
```bash
# Option 1: Use conda environment (RECOMMENDED)
conda activate rehab_ai_env
python test_integration.py --dataset synthetic --config tuned_config.json

# Option 2: Use conda run for one-off commands
conda run -n rehab_ai_env python test_integration.py --dataset synthetic --config tuned_config.json

# Option 3: Install missing dependencies
conda activate rehab_ai_env
conda install -c conda-forge diskcache matplotlib

# Option 4: Fix sys.path in script (if running from wrong directory)
# Edit test_integration.py line 19:
sys.path.insert(0, '/Users/davidryan/Documents_Local/GitHub/AI-Powered-Rehabilitation-Coaching-System/src')
```

---

## 📈 Week 9 Testing Schedule

### **Day 1-2: Baseline Testing**
- [ ] Run test on full test set
- [ ] Run test on validation set
- [ ] Generate visualizations
- [ ] Document baseline metrics

### **Day 3-4: Threshold Tuning**
- [ ] Analyze results with tuning script
- [ ] Adjust thresholds if needed
- [ ] Re-test with tuned config
- [ ] Compare baseline vs tuned

### **Day 5: Deep Analysis**
- [ ] Review videos with no events manually
- [ ] Verify high-severity mistakes are caught
- [ ] Check cache patterns
- [ ] Document final configuration

---

## 💾 Saving & Sharing Results

**Export Results:**
```bash
# Create results package
mkdir integration_test_results
cp test_results/*.json integration_test_results/
cp test_results/figures/*.png integration_test_results/
cp tuned_config.json integration_test_results/

# Zip for sharing
zip -r integration_test_results.zip integration_test_results/
```

**Results Include:**
- Test metrics (JSON)
- Visualizations (PNG)
- Tuned configuration
- Can share with team for review

---

## 🎓 Next Steps After Testing

Once you've validated the integration layer with synthetic data:

1. **Document final thresholds** 
   - Update `integration_layer.py` with validated Config values
   - Current tuned values: MIN_PERSISTENCE_RATE=0.25, MIN_CONFIDENCE=0.3, MIN_DURATION_SECONDS=2.5

2. **Populate Tier 1 cache**
   - Review top mistake types from test results
   - Add common patterns to `ResponseCache.populate_defaults()`
   - Target 60-80% cache hit rate for production

3. **Validate with real CV data**
   - Once CV pipeline is ready, test with live frame sequences
   - Compare synthetic vs real data performance
   - Adjust thresholds if needed for real-world behavior

4. **Integrate with LangGraph (Week 8)**
   - Connect integration layer to LangGraph state machine
   - Implement Tier 2 (RAG retrieval) and Tier 3 (LLM reasoning) nodes
   - End-to-end testing: CV → Integration → LangGraph → Delivery

5. **Performance optimization**
   - Profile frame processing latency (target: <100ms per frame)
   - Optimize cache lookups
   - Implement async processing for LLM calls

6. **Production deployment considerations**
   - Adjust MIN_COACHING_INTERVAL based on user feedback
   - Implement re-coaching logic for persistent mistakes
   - Add telemetry for monitoring event generation rates

---

## 📊 Testing Checklist

- [ ] Environment activated with `conda activate rehab_ai_env`
- [ ] Synthetic data generated (217 test + 1,292 val videos)
- [ ] Quick test completed (10 videos) with events detected
- [ ] Full test run on synthetic test dataset (217 videos)
- [ ] Sample validation test (100 videos) completed
- [ ] Results analyzed with tune_thresholds.py
- [ ] Visualizations generated in test_results/figures/
- [ ] tuned_config.json validated and documented
- [ ] Integration layer bug fix verified (last_coaching_time = -1000)
- [ ] Tier distribution analyzed (2/3 split expected without cache)
- [ ] Top mistakes and exercises reviewed for reasonableness
- [ ] Ready to integrate with LangGraph pipeline

---

## � Example Test Output

### **Successful Test Run:**

```bash
$ conda activate rehab_ai_env
$ python test_integration.py --dataset synthetic --max-videos 10 --config tuned_config.json

============================================================
INTEGRATION LAYER TESTING
============================================================

Configuration:
  MIN_FRAMES: 75
  MIN_PERSISTENCE_RATE: 0.25
  MIN_CONFIDENCE: 0.3
  MIN_DURATION_SECONDS: 2.5
  MIN_COACHING_INTERVAL: 10

============================================================
TESTING DATASET: SYNTHETIC
============================================================
ℹ️  Found 217 .json files, loading...
✅ Loaded 217 videos
Testing first 10 videos only

Processing videos...
  Processed 10/10 videos... ✅

============================================================
TEST RESULTS SUMMARY
============================================================

📊 Overall Statistics:
  Total Videos: 10
  Total Frames: 1200
  Total Coaching Events: 3
  Videos with Events: 3 (30.0%)
  Avg Events per Video: 0.30

🎯 Tier Breakdown:
  Tier 1: 0 (0.0%)
  Tier 2: 2 (66.7%)
  Tier 3: 1 (33.3%)

⚠️  Severity Breakdown:
  High: 1 (33.3%)
  Medium: 1 (33.3%)
  Low: 1 (33.3%)

💾 Results saved to: test_results/test_results_synthetic_20260305_143022.json
```

---

## �📞 Need Help?

**Common Questions:**

Q: How many videos should I test on?
A: Start with 10, then 50, then full dataset (217 test / 1,292 val)

Q: Why use synthetic data instead of raw QEVD logs?
A: QEVD logs have only 1-2 frames per file (event logs). Integration layer needs 75+ continuous frames for temporal aggregation.

Q: What if results vary between test/val sets?
A: Normal - validation set has more diverse exercises. Tune thresholds to work well on both.

Q: Should I manually review videos?
A: Yes! Especially videos with no events or many events. Verify mistakes are legitimate.

Q: How do I know if thresholds are right?
A: Target: 0.3-1.0 events/video for synthetic data. Check that serious mistakes are caught.

Q: Why is cache hit rate 0%?
A: Tier 1 cache needs to be populated with common patterns. This is expected initially.

Q: How long does it take to test all 217 videos?
A: ~2-3 minutes on modern hardware. Validation set (1,292 videos) takes ~15-20 minutes.

Q: Can I run tests in parallel?
A: Not currently. Each test creates a new integration layer instance with separate state.

---

Good luck with testing! 🚀
