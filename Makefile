# Makefile for SGD in C++ vs. scikit-learn SGDRegressor Reproduction
#
# Enforces single-threaded execution and strict failure propagation.

SHELL := bash
.SHELLFLAGS := -euo pipefail -c

ifeq ($(OS),Windows_NT)
    PYTHON := .venv\Scripts\python.exe
    EXE_EXT := .exe
    BIN_DIR := build
    CPP_TARGET := $(BIN_DIR)\sgd_cpp$(EXE_EXT)
else
    PYTHON := .venv/bin/python
    EXE_EXT :=
    BIN_DIR := build
    CPP_TARGET := $(BIN_DIR)/sgd_cpp$(EXE_EXT)
endif

# C++ Compiler and flags
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Icpp/include
CPP_TARGET := build/sgd_cpp$(EXE_EXT)

# Concurrency environment flags (single-threaded for benchmarking)
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1
export NUMEXPR_NUM_THREADS := 1

.PHONY: all setup build test data data-quick smoke pilot rq1 rq2 report verify clean distclean

all: setup build test data rq1 rq2 report

setup:
	@echo "==> Setting up environment..."
	@$(PYTHON) -m sgdrepro.envinfo

build: $(CPP_TARGET)

$(CPP_TARGET): cpp/src/main.cpp $(wildcard cpp/include/*.hpp)
	@echo "==> Compiling C++ binary: $@"
	@$(MKDIR_P) build
	$(CXX) $(CXXFLAGS) -o $(CPP_TARGET) cpp/src/main.cpp

test: build
	@echo "==> Running pytest test suite..."
	@$(PYTHON) -m pytest tests -v

data:
	@echo "==> Generating RQ1 synthetic datasets and preprocessing RQ2 housing datasets..."
	@$(PYTHON) -m sgdrepro.gen_rq1
	@$(PYTHON) -m sgdrepro.prep_rq2

data-quick:
	@echo "==> Generating small-scale quick datasets for smoke testing..."
	@$(PYTHON) -m sgdrepro.gen_rq1 --quick
	@$(PYTHON) -m sgdrepro.prep_rq2 --quick

smoke: build data-quick test
	@echo "==> Running smoke benchmarks (minimal grid, output to results_smoke/)..."
	@$(PYTHON) -m sgdrepro.run_rq1 --quick --out-dir results_smoke/raw/rq1
	@$(PYTHON) -m sgdrepro.run_rq2 --quick --out-dir results_smoke/raw/rq2
	@echo "==> Smoke test passed successfully."

pilot: build
	@echo "==> Running pilot noise and variance calibration..."
	@$(PYTHON) -m sgdrepro.run_rq1 --pilot

rq1: build
	@echo "==> Running RQ1 benchmark (Grids A, B, C)..."
	@$(PYTHON) -m sgdrepro.run_rq1

rq2: build
	@echo "==> Running RQ2 benchmark (Housing datasets & convergence)..."
	@$(PYTHON) -m sgdrepro.run_rq2

report:
	@echo "==> Generating result tables, figures, and summary markdown..."
	@$(PYTHON) -m sgdrepro.make_tables
	@$(PYTHON) -m sgdrepro.make_figures
	@$(PYTHON) -m sgdrepro.summary_gen

verify:
	@echo "==> Verifying benchmark results against reference..."
	@$(PYTHON) -m sgdrepro.compare_reference

clean:
	@echo "==> Cleaning build artifacts and cache..."
	@$(RM_RF) build results_smoke .pytest_cache

distclean: clean
	@echo "==> Removing generated data, results, and logs..."
	@$(RM_RF) data results logs
