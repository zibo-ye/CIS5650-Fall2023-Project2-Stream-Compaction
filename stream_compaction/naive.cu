#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(int N, int offset, int* odata, const int* idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= N)
            {
                return;
            }
            if (index >= offset)
            {
                odata[index] = idata[index - offset] + idata[index];
			}
            else
            {
				odata[index] = idata[index];
			}
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int loopTime = ilog2ceil(n);
            int* tmpIn, * tmpOut;
            cudaMalloc((void**)&tmpIn, sizeof(int) * n);
            cudaMalloc((void**)&tmpOut, sizeof(int) * n);
            cudaMemcpy(tmpIn, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            const int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            for (int i = 0; i < loopTime; i++)
            {
                kernNaiveScan CUDA_KERNEL(fullBlocksPerGrid, blockSize) (n, 1<<i, tmpOut, tmpIn);
                std::swap(tmpIn, tmpOut);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata+1, tmpIn, sizeof(int) * (n-1), cudaMemcpyDeviceToHost);
            odata[0] = 0;
            cudaFree(tmpIn);
            cudaFree(tmpOut);
        }
    }
}
