ARCH ?=gfx1030
ROCM_PATH ?=/opt/rocm
ROCPROF_LIBS ?=$(ROCM_PATH)/lib
ROCM_INCLUDES=-I$(ROCM_PATH)/include
ROCPROFILER_INCLUDES=-I$(ROCPROF_LIBS)/../include/rocprofiler/


CXX      = $(ROCM_PATH)/bin/hipcc
LDFLAGS  = -L$(ROCPROF_LIBS)/lib -lrocprofiler64 \
		   -L$(ROCM_PATH)/lib -lhsa-runtime64 -ldl
CXXFLAGS = --offload-arch=$(ARCH) -g -O0

all: standalone_v1_poll

standalone_v1_poll: standalone_v1_poll.cpp
	$(CXX) $^ $(LDFLAGS) -o $@ $(ROCPROFILER_INCLUDES) -I$(ROCM_PATH)/include/hsa $(CFLAGS)


clean:
	rm -f standalone_v1_poll

