#include <assert.h>
#include <hip/hip_runtime.h>
#include <hsa.h>
#include <rocprofiler.h>
#include <stdlib.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <vector>

// NOTE: Edit the metrics here
// look in metrics.xml for the list of available metrics
static const std::vector<const char*> metrics = {
    "ACTIVE_CYCLES",  // this one is available on most devices
    "ELAPSED_CYCLES", "ACTIVE_WAVES", "MeanOccupancyPerCU", "TOTAL_16_OPS",
    "TOTAL_32_OPS",   "TOTAL_64_OPS", "FETCH_SIZE",         "WRITE_SIZE"};

// Read all metrics LOOP_COUNT amount of times
// NOTE: Change this value to the number of times you want to read the metrics
static const uint32_t LOOP_COUNT = 10;
// Inner loop sleep
static const uint32_t METRIC_SLEEP = 1000 * 1;
// Outer loop sleep
static const uint32_t LOOP_SLEEP = 1000 * 50;

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
  hsa_agent_t* agents;
  unsigned count;
  unsigned capacity;
} hsa_agent_arr_t;

hsa_status_t get_agent_handle_cb(hsa_agent_t agent, void* agent_arr) {
  hsa_device_type_t type;
  hsa_agent_arr_t* agent_arr_ = (hsa_agent_arr_t*)agent_arr;

  hsa_status_t hsa_errno = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (hsa_errno != HSA_STATUS_SUCCESS) {
    return hsa_errno;
  }

  if (type == HSA_DEVICE_TYPE_GPU) {
    if (agent_arr_->count >= agent_arr_->capacity) {
      agent_arr_->capacity *= 2;
      agent_arr_->agents =
          (hsa_agent_t*)realloc(agent_arr_->agents, agent_arr_->capacity * sizeof(hsa_agent_t));
      assert(agent_arr_->agents);
    }
    agent_arr_->agents[agent_arr_->count] = agent;
    ++agent_arr_->count;
  }

  return HSA_STATUS_SUCCESS;
}

int get_agents(hsa_agent_arr_t* agent_arr) {
  int errcode = 0;
  hsa_status_t hsa_errno = HSA_STATUS_SUCCESS;

  agent_arr->capacity = 1;
  agent_arr->count = 0;
  agent_arr->agents = (hsa_agent_t*)calloc(agent_arr->capacity, sizeof(hsa_agent_t));
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

void read_features(rocprofiler_t* context, rocprofiler_feature_t* features,
                   const unsigned feature_count) {
  hsa_status_t hsa_errno = rocprofiler_read(context, 0);
  assert(hsa_errno == HSA_STATUS_SUCCESS);
  hsa_errno = rocprofiler_get_data(context, 0);
  assert(hsa_errno == HSA_STATUS_SUCCESS);
  hsa_errno = rocprofiler_get_metrics(context);
  assert(hsa_errno == HSA_STATUS_SUCCESS);
  // std::cout << "FEATURES:" << std::endl;
  for (int i = 0; i < feature_count; i++) {
    if (features[i].data.kind == ROCPROFILER_DATA_KIND_DOUBLE) {
      std::cout << "[" << features[i].data.result_double << "]\n";
    } else {
      std::cout << "Unsupported data type: " << features[i].data.kind << "\n";
    }
  }
}

void setup_profiler_env() {
  // set path to metrics.xml
  // setenv("ROCP_METRICS", "/opt/rocm/lib/rocprofiler/metrics.xml", 0);
  setenv("ROCP_METRICS", "./metrics.xml", 0);
}

#define QUEUE_NUM_PACKETS 64

bool createHsaQueue(hsa_queue_t** queue, hsa_agent_t gpu_agent) {
  // create a single-producer queue
  // TODO: check if API args are correct, especially UINT32_MAX
  hsa_status_t status;
  status = hsa_queue_create(gpu_agent, QUEUE_NUM_PACKETS, HSA_QUEUE_TYPE_SINGLE, NULL, NULL,
                            UINT32_MAX, UINT32_MAX, queue);
  if (status != HSA_STATUS_SUCCESS) fprintf(stdout, "Queue creation failed");

  // TODO: warning: is it really required!! ??
  status = hsa_amd_queue_set_priority(*queue, HSA_AMD_QUEUE_PRIORITY_HIGH);
  if (status != HSA_STATUS_SUCCESS) fprintf(stdout, "HSA Queue Priority Set Failed");

  return (status == HSA_STATUS_SUCCESS);
}

int run_profiler(const char* feature_name, hsa_agent_arr_t agent_arr, hsa_queue_t** queues) {
  const int features_count = 1;
  const char* events[features_count] = {feature_name};
  rocprofiler_feature_t features[MAX_DEV_COUNT][features_count];

  // initialize hsa. hsa_init() will also load the profiler libs under the hood
  hsa_status_t hsa_errno = HSA_STATUS_SUCCESS;

  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    for (int j = 0; j < features_count; ++j) {
      features[i][j].kind = (rocprofiler_feature_kind_t)ROCPROFILER_FEATURE_KIND_METRIC;
      features[i][j].name = events[j];
    }
  }

  rocprofiler_t* contexts[MAX_DEV_COUNT] = {0};
  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    rocprofiler_properties_t properties = {
        queues[i],
        64,
        NULL,
        NULL,
    };
    int mode = (ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_SINGLEGROUP);
    hsa_errno = rocprofiler_open(agent_arr.agents[i], features[i], features_count, &contexts[i],
                                 mode, &properties);
    const char* error_string;
    rocprofiler_error_string(&error_string);
    if (error_string != NULL) {
      fprintf(stdout, "%s", error_string);
      fflush(stdout);
    }
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    hsa_errno = rocprofiler_start(contexts[i], 0);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  // this is the duration for which the counter increments from zero.
  usleep(1000);

  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    hsa_errno = rocprofiler_stop(contexts[i], 0);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    // printf("Iteration %d\n", loopcount++);
    // fprintf(stdout, "------ Collecting Device[%d] -------\n", i);
    read_features(contexts[i], features[i], features_count);
  }

  usleep(100);

  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    hsa_errno = rocprofiler_close(contexts[i]);
    assert(hsa_errno == HSA_STATUS_SUCCESS);
  }

  return 0;
}

int main() {
  std::signal(SIGINT, signal_handler);

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

  // populate list of agents
  hsa_agent_arr_t agent_arr;
  int errcode = get_agents(&agent_arr);
  if (errcode != 0) {
    return -1;
  }

  hsa_queue_t* queues[MAX_DEV_COUNT];
  for (int i = 0; i < MAX_DEV_COUNT; ++i) {
    if (!createHsaQueue(&queues[i], agent_arr.agents[i]))
      fprintf(stdout, "can't create queues[%d]\n", i);
  }

  const auto start_time = std::chrono::system_clock::now();

  // run profiler
  for (int i = 0; (i < LOOP_COUNT) && (!signalled); i++) {
    printf("------ [%04d] ------\n", i);
    for (const auto& metric : metrics) {
      printf("%-20s", metric);
      status = run_profiler(metric, agent_arr, queues);
      assert(status == 0);
      usleep(METRIC_SLEEP);
    }
    usleep(LOOP_SLEEP);
  }

  if (!signalled) {
    // calculate end time
    const auto end_time = std::chrono::system_clock::now();
    const auto elapsed_time = end_time - start_time;
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time);
    const auto duration_no_sleep =
        duration.count() - ((metrics.size() * METRIC_SLEEP + LOOP_SLEEP) * LOOP_COUNT);
    const auto per_loop = duration_no_sleep / LOOP_COUNT;
    const auto per_metric = per_loop / metrics.size();
    std::cout << "Execution time: \n";
    std::cout << "    per_loop: " << per_loop << " us\n";
    std::cout << "    per_metric: " << per_metric << " us\n";
  }

  free(agent_arr.agents);

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
