# GPU_M2D G3 timing probe build.
#
# The -O0 backend flags are required: with either compiler pipeline above
# -O0 the measured PTX loads can be optimized away (see docs/G3_SURVEY.md
# section 2, compiler-flags note from the reference tooling). The pool
# harness shares those flags because it times with the same kernels.

NVCC ?= nvcc
CUDA_ARCH ?= 89
NVCCFLAGS := -std=c++17 -arch=sm_$(CUDA_ARCH) -O3 -Xcicc -O0 -Xptxas -O0 \
	--generate-line-info

.PHONY: all clean check

all: g3_timing_probe g3_pool_harness

g3_timing_probe: g3_timing_probe.cu g3_timing_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

g3_pool_harness: g3_pool_harness.cu g3_timing_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lcuda

check: all
	python3 -m py_compile analyze_scan.py analyze_bit_scan.py \
		analyze_pair_scan.py analyze_local_recal.py analyze_census.py \
		analyze_three_band.py analyze_channel_graph.py solve_mapping.py \
		build_bank_table.py analyze_table_build.py g3_pool.py
	python3 analyze_scan.py --self-test
	python3 analyze_bit_scan.py --self-test
	python3 analyze_pair_scan.py --self-test
	python3 analyze_local_recal.py --self-test
	python3 analyze_census.py --self-test
	python3 analyze_three_band.py --self-test
	python3 analyze_channel_graph.py --self-test
	python3 solve_mapping.py --self-test
	python3 build_bank_table.py --self-test
	python3 analyze_table_build.py --self-test
	python3 g3_pool.py --self-test

clean:
	rm -f g3_timing_probe g3_pool_harness __pycache__/*.pyc
