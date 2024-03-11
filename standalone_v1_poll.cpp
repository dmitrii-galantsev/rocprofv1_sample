#include <assert.h>
#include <hip/hip_runtime.h>
#include <hsa.h>
#include <atomic>
#include <cstdio>
#include <iostream>
#include <rocprofiler.h>
#include <stdlib.h>
#include <unistd.h>
#include <csignal>
#include <vector>


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

#define MAX_DEV_COUNT (1)

using namespace std;

typedef struct {
  hsa_agent_t *agents;
  unsigned count;
  unsigned capacity;
} hsa_agent_arr_t;

hsa_status_t get_agent_handle_cb(hsa_agent_t agent, void *agent_arr) {
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
      assert(agent_arr_->agents);
    }
    agent_arr_->agents[agent_arr_->count] = agent;
    ++agent_arr_->count;
  }

  return HSA_STATUS_SUCCESS;
}

int get_agents(hsa_agent_arr_t *agent_arr) {
  int errcode = 0;
  hsa_status_t hsa_errno = HSA_STATUS_SUCCESS;

  agent_arr->capacity = 1;
  agent_arr->count = 0;
  agent_arr->agents =
      (hsa_agent_t *)calloc(agent_arr->capacity, sizeof(hsa_agent_t));
  assert(agent_arr->agents);

  hsa_errno = hsa_iterate_agents(get_agent_handle_cb, agent_arr);
  if (hsa_errno != HSA_STATUS_SUCCESS) {
    errcode = -1;
    agent_arr->capacity = 0;
    agent_arr->count = 0;
    free(agent_arr->agents);
  }
  return errcode;
}

std::atomic_bool signalled = false;

void signal_handler(int signal) {
  printf("Signal received\n");
  std::flush(std::cout);
  signalled = true;
}

void read_features(rocprofiler_t *context, rocprofiler_feature_t *features,
                   const unsigned feature_count) {
  hsa_status_t hsa_errno = rocprofiler_read(context, 0);
  assert(hsa_errno == HSA_STATUS_SUCCESS);
  hsa_errno = rocprofiler_get_data(context, 0);
  assert(hsa_errno == HSA_STATUS_SUCCESS);
  hsa_errno = rocprofiler_get_metrics(context);
  assert(hsa_errno == HSA_STATUS_SUCCESS);
  //std::cout << "FEATURES:" << std::endl;
  for (int i = 0; i < feature_count; i++) {
    if (features[i].data.kind == ROCPROFILER_DATA_KIND_DOUBLE) {
      std::cout << "[" << features[i].data.result_double << "]\n";
    } else {
      std::cout << "Weird data type: " << features[i].data.kind << "\n";
    }
  }
}

void setup_profiler_env() {
  // set path to metrics.xml
  //setenv("ROCP_METRICS", "/opt/rocm/lib/rocprofiler/metrics.xml", 0);
  setenv("ROCP_METRICS", "/opt/rocm/libexec/rocprofiler/counters/derived_counters.xml", 0);
}

#define QUEUE_NUM_PACKETS 64

int run_profiler(const char * feature_name) {
  // populate list of agents
  hsa_agent_arr_t agent_arr;
  int errcode = get_agents(&agent_arr);
  if (errcode != 0) {
    return -1;
  }
  const int features_count = 1;
  const char *events[features_count] = {feature_name};
  rocprofiler_feature_t features[MAX_DEV_COUNT][features_count];

  // initialize hsa. hsa_init() will also load the profiler libs under the hood
  hsa_status_t hsa_errno = HSA_STATUS_SUCCESS;

  hsa_queue_t *queues[MAX_DEV_COUNT];

  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    for (int j = 0; j < features_count; ++j) {
      features[i][j].kind =
          (rocprofiler_feature_kind_t)ROCPROFILER_FEATURE_KIND_METRIC;
      features[i][j].name = events[j];
    }
  }

  rocprofiler_t *contexts[MAX_DEV_COUNT] = {0};
  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    rocprofiler_properties_t properties = {
        queues[i],
        64,
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
    if (error_string != NULL) {
      fprintf(stdout, "%s\n", error_string);
      fflush(stdout);
    }
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    hsa_errno = rocprofiler_start(contexts[i], 0);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  int loopcount = 0;

  //while (!signalled) {
    usleep(1000);
    for (int i = 0; i < MAX_DEV_COUNT; ++i) {
      //printf("Iteration %d\n", loopcount++);
      //fprintf(stdout, "------ Collecting Device[%d] -------\n", i);
      read_features(contexts[i], features[i], features_count);
    }
  //}

  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    hsa_errno = rocprofiler_stop(contexts[i], 0);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  usleep(10);

  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    hsa_errno = rocprofiler_close(contexts[i]);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  free(agent_arr.agents);

  return 0;
}

int main() {
  std::signal(SIGINT, signal_handler);
  std::vector<const char *> metrics = {
    "TA_BUSY_avr",
    //"CU_OCCUPANCY",
    //"CU_UTILIZATION",
    //"TA_UTIL",
    //"GDS_UTIL",
    //"EA_UTIL",
  };

  // setup
  setup_profiler_env();
  int status;
  auto hsa_errno = hsa_init();
  if (hsa_errno != HSA_STATUS_SUCCESS) {
    printf("hsa_init() failed! %d\n", hsa_errno);
    if (hsa_errno == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
      // NOTE: this issue is possibly due to rocprofiler not cleaning up the queues
      printf("ERR: HSA_STATUS_ERROR_OUT_OF_RESOURCES\n");
    }
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  // run profiler
  for (int i = 0; i < 300; i++) {
    printf("------ [%03d] ------\n", i);
    for (const auto &metric : metrics) {
      printf("%-20s", metric);
      status = run_profiler(metric);
      assert(status == 0);
      usleep(1000 * 1);
    }
    usleep(1000 * 5);
  }

  // break down
  hsa_errno = hsa_shut_down();
  if (hsa_errno != HSA_STATUS_SUCCESS) {
    printf("hsa_shut_down() failed! %d\n", hsa_errno);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }
  hsa_errno = hsa_shut_down();
  if (hsa_errno != HSA_STATUS_ERROR_NOT_INITIALIZED) {
    printf("hsa_shut_down() failed! %d\n", hsa_errno);
    assert(hsa_errno == HSA_STATUS_ERROR_NOT_INITIALIZED);
  }
  return 0;
}
