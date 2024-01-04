#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            thrust::device_vector<int> dev_thrust_vec_in(n);
            thrust::device_vector<int> dev_thrust_vec_out(n);

            thrust::copy(idata, idata + n, dev_thrust_vec_in.begin());

            timer().startGpuTimer();
            thrust::exclusive_scan(dev_thrust_vec_in.begin(), dev_thrust_vec_in.end(), dev_thrust_vec_out.begin());
            timer().endGpuTimer();

            thrust::copy(dev_thrust_vec_out.begin(), dev_thrust_vec_out.end(), odata);
        }
    }
}
