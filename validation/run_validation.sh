#!/bin/bash
# =============================================================
# run_validation.sh
# Master validation script for Nebius H200 cluster
# Runs all 10 validation tests in sequence and produces
# a clean PASS/FAIL report for documentation and demo day
# =============================================================

# ── Configuration ─────────────────────────────────────────────
VENV_PATH="$HOME/nebius-env"
TESTS_DIR="$(dirname "$0")/tests"
RESULTS_DIR="$(dirname "$0")/../results"
REPORT_FILE="$RESULTS_DIR/validation_report_$(date +%Y%m%d_%H%M%S).txt"
NODE_HOSTNAME=$(hostname)
NODE_RANK=${RANK:-0}

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # no color

# ── Helper Functions ───────────────────────────────────────────
print_header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║         NEBIUS H200 CLUSTER VALIDATION SUITE         ║"
    echo "║                                                      ║"
    echo "║  Node:     $(printf '%-40s' "$NODE_HOSTNAME")║"
    echo "║  Rank:     $(printf '%-40s' "$NODE_RANK")║"
    echo "║  Time:     $(printf '%-40s' "$(date '+%Y-%m-%d %H:%M:%S UTC')")║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""
}

print_section() {
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "  Running: $1"
    echo "──────────────────────────────────────────────────────"
}

print_result() {
    local test_name=$1
    local exit_code=$2

    if [ $exit_code -eq 0 ]; then
        echo -e "  ${GREEN}[PASS]${NC} $test_name"
    else
        echo -e "  ${RED}[FAIL]${NC} $test_name"
    fi
}

print_summary() {
    local total=$1
    local passed=$2
    local failed=$3

    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║              VALIDATION SUMMARY REPORT               ║"
    echo "╠══════════════════════════════════════════════════════╣"
    echo "║  Node: $(printf '%-46s' "$NODE_HOSTNAME")║"
    echo "╠══════════════════════════════════════════════════════╣"

    # Print each test result
    for i in "${!TEST_NAMES[@]}"; do
        local name="${TEST_NAMES[$i]}"
        local code="${TEST_RESULTS[$i]}"
        if [ "$code" -eq 0 ]; then
            status="PASS ✓"
        else
            status="FAIL ✗"
        fi
        echo "║  $(printf '%-38s' "$name") $(printf '%-13s' "$status")║"
    done

    echo "╠══════════════════════════════════════════════════════╣"
    echo "║  Total:  $total tests                                        ║"
    echo "║  Passed: $passed tests                                        ║"
    echo "║  Failed: $failed tests                                        ║"
    echo "╠══════════════════════════════════════════════════════╣"

    if [ $failed -eq 0 ]; then
        echo "║                                                      ║"
        echo "║        ALL TESTS PASSED — CLUSTER IS HEALTHY         ║"
        echo "║          Ready for training workloads ✓              ║"
        echo "║                                                      ║"
    else
        echo "║                                                      ║"
        echo "║        $failed TEST(S) FAILED — REVIEW BEFORE TRAINING    ║"
        echo "║          Check logs above for failure details        ║"
        echo "║                                                      ║"
    fi

    echo "╚══════════════════════════════════════════════════════╝"
    echo ""
}

# ── Environment Setup ──────────────────────────────────────────
setup_environment() {
    echo ""
    echo "--- Environment Setup ---"

    # Activate virtual environment
    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
        echo "  [PASS] Virtual environment activated: $VENV_PATH"
    else
        echo "  [FAIL] Virtual environment not found at $VENV_PATH"
        echo "  [INFO] Run: python3 -m venv $VENV_PATH && pip install -r requirements.txt"
        exit 1
    fi

    # Verify Python is from venv
    PYTHON_PATH=$(which python3)
    echo "  [INFO] Python: $PYTHON_PATH"

    # Create results directory if it doesn't exist
    mkdir -p "$RESULTS_DIR"
    echo "  [INFO] Results will be saved to: $RESULTS_DIR"

    # Verify tests directory exists
    if [ ! -d "$TESTS_DIR" ]; then
        echo "  [FAIL] Tests directory not found: $TESTS_DIR"
        exit 1
    fi

    echo "  [INFO] Tests directory: $TESTS_DIR"
    echo "  [INFO] Node: $NODE_HOSTNAME (rank $NODE_RANK)"
}

# ── Run a Single Test ──────────────────────────────────────────
run_test() {
    local test_num=$1
    local test_name=$2
    local test_file=$3
    local test_type=$4  # "python" or "bash"

    print_section "$test_num: $test_name"

    # Check test file exists
    if [ ! -f "$TESTS_DIR/$test_file" ]; then
        echo "  [FAIL] Test file not found: $TESTS_DIR/$test_file"
        return 1
    fi

    # Run the test
    if [ "$test_type" = "python" ]; then
        python3 "$TESTS_DIR/$test_file"
    else
        bash "$TESTS_DIR/$test_file"
    fi

    return $?
}

# ── Main Execution ─────────────────────────────────────────────
main() {

    # Print header
    print_header

    # Setup environment
    setup_environment

    # Arrays to track results
    TEST_NAMES=()
    TEST_RESULTS=()

    echo ""
    echo "Starting validation suite on node: $NODE_HOSTNAME"
    echo "$(date '+%Y-%m-%d %H:%M:%S UTC')"
    echo ""

    # ── Test 01: GPU Health ────────────────────────────
    run_test "01" "GPU Health Check" "01_gpu_health.py" "python"
    TEST_RESULTS+=($?)
    TEST_NAMES+=("01 GPU Health")

    # ── Test 02: CUDA Functionality ────────────────────
    run_test "02" "CUDA Functionality" "02_cuda_check.py" "python"
    TEST_RESULTS+=($?)
    TEST_NAMES+=("02 CUDA Functionality")

    # ── Test 03: NCCL Intra-Node ───────────────────────
    run_test "03" "NCCL Intra-Node Bandwidth" "03_nccl_intranode.sh" "bash"
    TEST_RESULTS+=($?)
    TEST_NAMES+=("03 NCCL Intra-Node")

    # ── Test 04: NCCL Inter-Node ───────────────────────
    run_test "04" "NCCL Inter-Node Bandwidth" "04_nccl_internode.sh" "bash"
    TEST_RESULTS+=($?)
    TEST_NAMES+=("04 NCCL Inter-Node")

    # ── Test 05: Storage Benchmark ─────────────────────
    run_test "05" "Storage & Filesystem" "05_storage_bench.py" "python"
    TEST_RESULTS+=($?)
    TEST_NAMES+=("05 Storage Benchmark")

    # ── Test 06: CPU & RAM ─────────────────────────────
    run_test "06" "CPU & RAM Check" "06_cpu_ram_check.py" "python"
    TEST_RESULTS+=($?)
    TEST_NAMES+=("06 CPU & RAM")

    # ── Test 07: Distributed Smoke Test ───────────────
    run_test "07" "Distributed Smoke Test" "07_distributed_smoke.py" "python"
    TEST_RESULTS+=($?)
    TEST_NAMES+=("07 Distributed Smoke")

    # ── Test 08: GPU Stress Test ───────────────────────
    run_test "08" "GPU Stress Test (30s)" "08_gpu_stress.py" "python"
    TEST_RESULTS+=($?)
    TEST_NAMES+=("08 GPU Stress Test")

    # ── Test 09: ECC Monitoring ────────────────────────
    run_test "09" "ECC & Xid Monitoring" "09_ecc_monitoring.py" "python"
    TEST_RESULTS+=($?)
    TEST_NAMES+=("09 ECC Monitoring")

    # ── Test 10: Environment Consistency ──────────────
    run_test "10" "Environment Consistency" "10_env_consistency.py" "python"
    TEST_RESULTS+=($?)
    TEST_NAMES+=("10 Environment Consistency")

    # ── Calculate Results ──────────────────────────────
    TOTAL=${#TEST_RESULTS[@]}
    PASSED=0
    FAILED=0

    for code in "${TEST_RESULTS[@]}"; do
        if [ "$code" -eq 0 ]; then
            PASSED=$((PASSED + 1))
        else
            FAILED=$((FAILED + 1))
        fi
    done

    # ── Print Summary ──────────────────────────────────
    print_summary $TOTAL $PASSED $FAILED

    # ── Save Report ────────────────────────────────────
    # Only rank 0 saves the report to avoid duplicate files
    if [ "$NODE_RANK" = "0" ]; then
        echo "Saving validation report to: $REPORT_FILE"

        {
            echo "NEBIUS H200 CLUSTER VALIDATION REPORT"
            echo "======================================"
            echo "Date:     $(date '+%Y-%m-%d %H:%M:%S UTC')"
            echo "Node:     $NODE_HOSTNAME"
            echo "Rank:     $NODE_RANK"
            echo ""
            echo "RESULTS:"
            for i in "${!TEST_NAMES[@]}"; do
                name="${TEST_NAMES[$i]}"
                code="${TEST_RESULTS[$i]}"
                if [ "$code" -eq 0 ]; then
                    echo "  $name: PASS"
                else
                    echo "  $name: FAIL"
                fi
            done
            echo ""
            echo "Total:  $TOTAL"
            echo "Passed: $PASSED"
            echo "Failed: $FAILED"
            echo ""
            if [ $FAILED -eq 0 ]; then
                echo "VERDICT: CLUSTER HEALTHY — READY FOR TRAINING"
            else
                echo "VERDICT: $FAILED FAILURE(S) — REVIEW BEFORE TRAINING"
            fi
        } > "$REPORT_FILE"

        echo "  [PASS] Report saved: $REPORT_FILE"
    fi

    # ── Exit Code ──────────────────────────────────────
    # Return 0 if all passed, 1 if any failed
    if [ $FAILED -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Run main
main 2>&1 | tee -a "$REPORT_FILE.tmp"
