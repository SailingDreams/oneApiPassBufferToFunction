//==============================================================
// This is used for debugging the passing of a sycl buffer to a function
// =============================================================
#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <windows.h> 

using namespace sycl;
using namespace std;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

/***************************************************************
 * 
 ****************************************************************/
//void initUint8SyclBuffer(queue &q,
//                      buffer<uint8_t, 1> &u8_buffer, // input and output
//                      int width, int height, uint8_t value)
void initUint8SyclBuffer(queue &q,
                      vector<uint8_t> &u8_vector, // input and output
                      int sz, uint8_t value)
{
  cout << "Running Sycl vector arg passing" << std::endl;
  range numItems{sz};

  buffer u8_buffer(u8_vector);
  try
  {  
      q.submit([&](auto &h) {
        accessor u8(u8_buffer, h, write_only, no_init);
        h.parallel_for(numItems, [=](auto idx) {
                          u8[idx] = value;
                      });
      });
  } catch (std::exception const &e) {
    cout << "initUint8SyclBuffer exception: " << e.what() << std::endl;
    terminate();
  }
}

/***************************************************************
 * 
 ****************************************************************/
void initUint8SyclBuffer1(queue &q,
                      buffer<uint8_t> &u8_buffer, // input and output
                      //uint8_t *u8_buffer, // input and output
                      int sz, uint8_t value)
{
  cout << "Running Sycl buffer arg passing" << std::endl;
  range numItems{sz};

  try
  {  
      //q.submit([&u8_buffer, width, height, value](
      //          sycl::handler& h) {
      q.submit([&](auto &h) {
      accessor u8(u8_buffer, h, write_only, no_init);
      h.parallel_for(numItems, [=](auto idx) {
                        u8[idx] = value;
                    });
      });
  } catch (std::exception const &e) {
    cout << "initUint8SyclBuffer exception: " << e.what() << std::endl;
    terminate();
  }
}

#define USING_CURLY_BRACKETS
//#undef USING_CURLY_BRACKETS
  
//**************************************************************************
// Demonstrate passing sycl buffers as arguments.
//**************************************************************************
int main() {
  // Create device selector for the device of your interest.
  // The default device selector will select the most performant device.
  auto selector = default_selector_v;
  cout << "Starting main" << std::endl;

  int sz = 10; 
  uint8_t initValue = 128;
  std::vector<uint8_t> u8_data(sz);
#ifdef USING_CURLY_BRACKETS
  { 
#endif  
    // Create sycl u8_data_buffer in this scope in order to make get the host memeory to 
    // sync with device memory when the u8_data_buffer goes out of scope.
    queue sycl_que(selector, exception_handler);
    buffer u8_data_buffer(u8_data);
    //buffer<uint8_t, 1> u8_data_buffer{u8_data.data(), range<1>(sz)};

    try {
      // Print out the device information used for the kernel code.
      cout << "Running on device: "
          << sycl_que.get_device().get_info<info::device::name>() << "\n";
      
      //initUint8SyclBuffer(sycl_que, u8_data, sz, initValue);    // passing vectors
      initUint8SyclBuffer1(sycl_que, u8_data_buffer, sz, initValue);  // passing sycl buffers

    } catch (std::exception const &e) {
      cout << "An exception is caught on device:  " << e.what() << std::endl;
      terminate();
    }

    sycl_que.wait();
#ifdef USING_CURLY_BRACKETS
  }  // Ending curly bracket takes sycl buffer out of scope and thus syncs it to host memory
  for(int i = 0; i < sz; i++) { 
      cout << (int)u8_data[i] << ", ";
#else
  host_accessor h_u8_data(u8_data_buffer,read_only);
  for(int i = 0; i < sz; i++) { 
      cout << (int)h_u8_data[i] << ", ";
#endif

      
#ifdef USING_CURLY_BRACKETS
      if(u8_data[i] != initValue) {
#else
      if(h_u8_data[i] != initValue) {
#endif
        cerr << "ERROR: u8_data[" << i << "] = " << (int)u8_data[i] << 
               " does not equal init value " << (int)initValue << "." << std::endl;
        exit(1);
      }
  }
  cout << std::endl;


  cout << "Success! The End.\n";
  return 0;
}
