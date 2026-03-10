#!/bin/bash

# Quick Start Testing Script
# Runs complete testing pipeline for Integration Layer

echo "=========================================="
echo "Integration Layer Testing - Quick Start"
echo "=========================================="
echo ""

# Check if running from correct directory
if [ ! -d "../../src/integration" ]; then
    echo "❌ Error: Please run this script from the tests/integration_testing/ directory"
    echo "   cd /path/to/repo/tests/integration_testing"
    exit 1
fi

# Create results directory
mkdir -p test_results/figures

echo "Step 1: Running baseline test (first 20 videos)..."
echo ""
python test_integration.py --dataset test --max-videos 20

# Get most recent results file
RESULTS_FILE=$(ls -t test_results/test_results_test_*.json | head -1)

if [ -z "$RESULTS_FILE" ]; then
    echo "❌ No results file found"
    exit 1
fi

echo ""
echo "✅ Test complete! Results: $RESULTS_FILE"
echo ""

echo "Step 2: Analyzing results and tuning thresholds..."
echo ""
python test_tune_thresholds.py --results "$RESULTS_FILE"

echo ""
echo "Step 3: Generating visualizations..."
echo ""
python test_visualize.py --results "$RESULTS_FILE"

echo ""
echo "=========================================="
echo "Quick Start Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Metrics: $RESULTS_FILE"
echo "  - Figures: test_results/figures/"
echo "  - Tuned config: tuned_config.json (if generated)"
echo ""
echo "Next steps:"
echo "  1. Review test_results/figures/ visualizations"
echo "  2. Check tuned_config.json for recommendations"
echo "  3. Run full test: python test_integration.py --dataset both"
echo ""
