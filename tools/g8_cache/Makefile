# GPU_M2D G8 L2-cache tooling build.
#
# The probe kernel lives in include/gpu_m2d/l2_probe_kernels.cuh and is
# shared with the G1.5 runner. Unlike the G3 DRAM timing probe it does not
# need -O0 backends: completion is forced by a data dependency (the loaded
# value is stored to shared memory before the second clock read), which
# the optimizer must respect.

NVCC ?= nvcc
CUDA_ARCH ?= 89
NVCCFLAGS := -std=c++17 -arch=sm_$(CUDA_ARCH) -O3 --generate-line-info \
	-I../../include

.PHONY: all clean check

all: g8_l2_calib

g8_l2_calib: g8_l2_calib.cu ../../include/gpu_m2d/l2_probe_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

check: all
	python3 -m py_compile analyze_l2_calib.py
	python3 analyze_l2_calib.py --self-test

clean:
	rm -f g8_l2_calib __pycache__/*.pyc
