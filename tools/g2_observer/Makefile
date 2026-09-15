# GPU_M2D G2 observer harness build.
#
# The eBPF observer tooling itself is Python (BCC) and is run with the
# system interpreter /usr/bin/python3, which is the only one exposing
# BCC on this host. Only the scratch harness is compiled here.

NVCC ?= nvcc
CUDA_ARCH ?= 89

.PHONY: all clean check

all: g2_scratch_harness g2_alias_harness

g2_scratch_harness: g2_scratch_harness.cu
	$(NVCC) -std=c++17 -arch=sm_$(CUDA_ARCH) -O2 -o $@ $< -lcuda

g2_alias_harness: g2_alias_harness.cu
	$(NVCC) -std=c++17 -arch=sm_$(CUDA_ARCH) -O2 -o $@ $< -lcuda

check: all
	/usr/bin/python3 -m py_compile g2_observer.py pte_decoder.py \
		run_g2_scratch_probe.py run_g2_tensorrt_probe.py run_g2_alias_probe.py \
		replay_trt_run.py va_pa_lookup.py
	/usr/bin/python3 g2_observer.py --self-check
	/usr/bin/python3 va_pa_lookup.py --self-test

clean:
	rm -f g2_scratch_harness g2_alias_harness __pycache__/*.pyc
