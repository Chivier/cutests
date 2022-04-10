#include <moderngpu/kernel_mergesort.hxx>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <chrono>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

using namespace mgpu;

int main(int argc, char** argv) {
  standard_context_t context;

  cudaDeviceSynchronize(); 
  auto begin_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  int count = 1000000;

    for(int it = 1; it <= 50; ++it) {
      mem_t<int> data = fill_random(0, 100000, count, false, context);
      
      mergesort(data.data(), count, less_t<int>(), context);

      std::vector<int> ref = from_mem(data);
      std::sort(ref.begin(), ref.end());
      std::vector<int> sorted = from_mem(data);
    }
  auto end_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  cudaDeviceSynchronize();
  printf("%ld\n", end_millis - begin_millis);

  cudaDeviceSynchronize(); 
  begin_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  count = 1000000;

    for(int it = 1; it <= 50; ++it) {
      mem_t<int> data = fill_random(0, 100000, count, false, context);
      
      launchboundsort(data.data(), count, less_t<int>(), context);

      std::vector<int> ref = from_mem(data);
      std::sort(ref.begin(), ref.end());
      std::vector<int> sorted = from_mem(data);
    }
  end_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  cudaDeviceSynchronize();
  printf("%ld\n", end_millis - begin_millis);
  return 0;
}

