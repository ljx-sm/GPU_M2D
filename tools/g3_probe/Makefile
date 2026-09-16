# GPU_M2D G3 timing probe build.
#
# The -O0 backend flags are required: with either compiler pipeline above
# -O0 the measured PTX loads can be optimized away (see docs/G3_SURVEY.md
# section 2, compiler-flags note from the reference tooling).

NVCC ?= nvcc
CUDA_ARCH ?= 89
NVCCFLAGS := -std=c++17 -arch=sm_$(CUDA_ARCH) -O3 -Xcicc -O0 -Xptxas -O0 \
	--generate-line-info

.PHONY: all clean check

all: g3_timing_probe

g3_timing_probe: g3_timing_probe.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

check: all
	python3 -m py_compile analyze_scan.py
	python3 analyze_scan.py --self-test

clean:
	rm -f g3_timing_probe __pycache__/*.pyc
