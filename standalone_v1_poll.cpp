#include <assert.h>
#include <hip/hip_runtime.h>
#include <hsa.h>
#include <iostream>
#include <rocprofiler.h>
#include <stdlib.h>
#include <unistd.h>
#include <csignal>

#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x) == hipSuccess))
#endif

#define WIDTH 1024
#define HEIGHT 1024

#define NUM (WIDTH * HEIGHT)

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define THREADS_PER_BLOCK_Z 1

#define MAX_DEV_COUNT (2)

using namespace std;

typedef struct {
  hsa_agent_t *agents;
  unsigned count;
  unsigned capacity;
} hsa_agent_arr_t;

static hsa_agent_arr_t agent_arr;

static hsa_status_t get_agent_handle_cb(hsa_agent_t agent, void *agent_arr) {
  hsa_device_type_t type;
  hsa_agent_arr_t *agent_arr_ = (hsa_agent_arr_t *)agent_arr;

  hsa_status_t hsa_errno =
      hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (hsa_errno != HSA_STATUS_SUCCESS) {
    return hsa_errno;
  }

  if (type == HSA_DEVICE_TYPE_GPU) {
    if (agent_arr_->count >= agent_arr_->capacity) {
      agent_arr_->capacity *= 2;
      agent_arr_->agents = (hsa_agent_t *)realloc(
          agent_arr_->agents, agent_arr_->capacity * sizeof(hsa_agent_t));
    }
    agent_arr_->agents[agent_arr_->count] = agent;
    ++agent_arr_->count;
  }

  return HSA_STATUS_SUCCESS;
}

static int get_agents(hsa_agent_arr_t *agent_arr) {
  int errcode = 0;
  hsa_status_t hsa_errno = HSA_STATUS_SUCCESS;

  agent_arr->capacity = 1;
  agent_arr->count = 0;
  agent_arr->agents =
      (hsa_agent_t *)calloc(agent_arr->capacity, sizeof(hsa_agent_t));
  assert(agent_arr->agents);

  hsa_errno = hsa_iterate_agents(get_agent_handle_cb, agent_arr);
  if (hsa_errno != HSA_STATUS_SUCCESS) {
    goto fn_fail;
  }

fn_exit:
  return errcode;
fn_fail:
  errcode = -1;

  agent_arr->capacity = 0;
  agent_arr->count = 0;
  free(agent_arr->agents);

  goto fn_exit;
}


bool signalled = false;

void signal_handler(int signal) {
  printf("Signal received\n");
  std::flush(std::cout);
  signalled = true;
}

void print_features(rocprofiler_feature_t *feature, uint32_t feature_count) {
  for (rocprofiler_feature_t *p = feature; p < feature + feature_count; ++p) {
    std::cout << (p - feature) << ": " << p->name;
    switch (p->data.kind) {
    case ROCPROFILER_DATA_KIND_INT64:
      std::cout << std::dec << " result64 (" << p->data.result_int64 << ")"
                << std::endl;
      break;
    case ROCPROFILER_DATA_KIND_DOUBLE:
      std::cout << " result64 (" << p->data.result_double << ")" << std::endl;
      break;
    case ROCPROFILER_DATA_KIND_BYTES: {
      const char *ptr =
          reinterpret_cast<const char *>(p->data.result_bytes.ptr);
      uint64_t size = 0;
      for (unsigned i = 0; i < p->data.result_bytes.instance_count; ++i) {
        size = *reinterpret_cast<const uint64_t *>(ptr);
        const char *data = ptr + sizeof(size);
        std::cout << std::endl;
        std::cout << std::hex << "  data (" << (void *)data << ")" << std::endl;
        std::cout << std::dec << "  size (" << size << ")" << std::endl;
        ptr = data + size;
      }
      break;
    }
    default:
      std::cout << "default!!!!!! "
                << "result kind (" << p->data.kind << ")" << std::endl;
      // TEST_ASSERT(false);
    }
  }
}

void read_features(rocprofiler_t *context, rocprofiler_feature_t *features,
                   const unsigned feature_count) {
  hsa_status_t hsa_errno = rocprofiler_read(context, 0);
  assert(hsa_errno == HSA_STATUS_SUCCESS);
  std::cout << "reading counters from hw" << std::endl;
  hsa_errno = rocprofiler_get_data(context, 0);
  assert(hsa_errno == HSA_STATUS_SUCCESS);
  std::cout << "evaluating metric expressions" << std::endl;
  hsa_errno = rocprofiler_get_metrics(context);
  assert(hsa_errno == HSA_STATUS_SUCCESS);
  std::cout << "RESULTS:" << std::endl;
  print_features(features, feature_count);
}

// This won't actually clear the features as the pmc register is the one which contains the counter value
// In order to do a proper clear, the pmc register also needs to be cleared.
// So the right approach would be to open/close a context each time after read.


void clear_features(rocprofiler_feature_t *feature, uint32_t feature_count) {
for (rocprofiler_feature_t *p = feature; p < feature + feature_count; ++p) {
    switch (p->data.kind) {
    case ROCPROFILER_DATA_KIND_INT64: {
      p->data.result_int64 = 0;
      printf("data.result_int64: %7lu\n", p->data.result_int64);
      break;
    }
    case ROCPROFILER_DATA_KIND_DOUBLE: {
      p->data.result_double = 0;
      printf("data.result_double: %3.3f\n", p->data.result_double);
      break;
    }
    default:
      std::cout << "default!!!!!! "
                << "result kind (" << p->data.kind << ")" << std::endl;
      // TEST_ASSERT(false);
    }
  }
}

void setup_profiler_env() {
  // set path to librocprofiler64.so.1
  // setenv(
  //     "HSA_TOOLS_LIB",
  //     "/home/sauverma/standalone_mode/rocprofiler/build/lib/librocprofiler64.so.1",
  //     0);
  // set path to metrics.xml
  setenv("ROCP_METRICS", "/opt/rocm/lib/rocprofiler/metrics.xml", 0);
}

#define QUEUE_NUM_PACKETS 64

bool createHsaQueue(hsa_queue_t **queue, hsa_agent_t gpu_agent) {
  // create a single-producer queue
  // TODO: check if API args are correct, especially UINT32_MAX
  hsa_status_t status;
  status = hsa_queue_create(gpu_agent, QUEUE_NUM_PACKETS, HSA_QUEUE_TYPE_SINGLE,
                            NULL, NULL, UINT32_MAX, UINT32_MAX, queue);
  if (status != HSA_STATUS_SUCCESS)
    fprintf(stdout, "Queue creation failed");

  // TODO: warning: is it really required!! ??
  status = hsa_amd_queue_set_priority(*queue, HSA_AMD_QUEUE_PRIORITY_HIGH);
  if (status != HSA_STATUS_SUCCESS)
    fprintf(stdout, "Device Profiling HSA Queue Priority Set Failed");

  return (status == HSA_STATUS_SUCCESS);
}

int main() {
  std::signal(SIGINT, signal_handler);

  setup_profiler_env();

  const int features_count = 2;
  const char *events[features_count] = {"GPU_UTIL", "GRBM_COUNT"};
  rocprofiler_feature_t features[MAX_DEV_COUNT][features_count];

  // initialize hsa. hsa_init() will also load the profiler libs under the hood
  hsa_status_t hsa_errno = HSA_STATUS_SUCCESS;
  hsa_errno = hsa_init();
  assert(hsa_errno == HSA_STATUS_SUCCESS);

  hsa_queue_t *queues[MAX_DEV_COUNT];

  // populate list of agents
  int errcode = get_agents(&agent_arr);
  if (errcode != 0) {
    return -1;
  }
  printf("number of devices: %u\n", agent_arr.count);
  printf("devices being profiled: %u\n", (int)MAX_DEV_COUNT);

  for (int i = 0; i < MAX_DEV_COUNT; ++i) {

    for (int j = 0; j < features_count; ++j) {
      features[i][j].kind =
          (rocprofiler_feature_kind_t)ROCPROFILER_FEATURE_KIND_METRIC;
      features[i][j].name = events[j];
    }
    // if (!createHsaQueue(&queues[i], agent_arr.agents[i]))
    //   fprintf(stdout, "can't create queues[%d]\n", i);
  }

  int sample_index = 0;
  int sample_count = 5;

  rocprofiler_t *contexts[MAX_DEV_COUNT] = {0};
  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    rocprofiler_properties_t properties = {
        queues[i],
        128,
        NULL,
        NULL,
    };
    int mode = (ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_CREATEQUEUE | ROCPROFILER_MODE_SINGLEGROUP);
    // int mode = (ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_SINGLEGROUP);
    hsa_errno =
        rocprofiler_open(agent_arr.agents[i], features[i], features_count,
                          &contexts[i], mode, &properties);
    const char *error_string;
    rocprofiler_error_string(&error_string);
    fprintf(stdout, "%s\n", error_string);
    fflush(stdout);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    hsa_errno = rocprofiler_start(contexts[i], 0);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  int loopcount = 0;

  while (!signalled) {

    for (int i = 0; i < MAX_DEV_COUNT; ++i) {
      printf("Iteration %d\n", loopcount++);
      fprintf(stdout, "------ Collecting Device[%d] -------\n", i);
      read_features(contexts[i], features[i], features_count);
      fprintf(stdout, "-------------------------\n\n");

    }

    sleep(1);

  }

  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    hsa_errno = rocprofiler_stop(contexts[i], 0);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    hsa_errno = rocprofiler_close(contexts[i]);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  return 0;
}
