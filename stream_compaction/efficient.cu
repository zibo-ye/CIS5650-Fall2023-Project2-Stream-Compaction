#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernEfficientScanUpSweep(int N, int offset, int* data)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= N)
            {
                return;
            }
            int idx_2 = (index + 1) * 2 * offset - 1;
            int idx_1 = idx_2 - offset;
            data[idx_2] += data[idx_1];
        }

        __global__ void kernEfficientScanDownSweep(int N, int offset, int* data)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= N)
            {
                return;
            }
            int idx_2 = (index + 1) * 2 * offset - 1;
            int idx_1 = idx_2 - offset;
            int t = data[idx_1];
            data[idx_1] = data[idx_2];
            data[idx_2] += t;
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            int loopTime = ilog2ceil(n);
            int N = (1 << loopTime);
            int* tmpIn;
            cudaMalloc((void**)&tmpIn, sizeof(int) * N);
            cudaMemcpy(tmpIn, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            
            timer().startGpuTimer();
            scan_impl(loopTime, tmpIn);
            timer().endGpuTimer();

            cudaMemcpy(odata, tmpIn, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(tmpIn);
        }

        void scan_impl(int loopTime, int* tmpIn) {
            int N = (1 << loopTime);
            const int blockSize = 128;
            for (int i = 0; i < loopTime; i++)
            {
                int workload = 1 << (loopTime - i - 1);
                dim3 fullBlocksPerGrid((workload + blockSize - 1) / blockSize);
                kernEfficientScanUpSweep CUDA_KERNEL(fullBlocksPerGrid, blockSize) (workload, 1 << i, tmpIn);
            }

            cudaMemset(tmpIn + N - 1, 0, sizeof(int)); // set the last element to 0

            for (int i = loopTime - 1; i >= 0; i--)
            {
                int workload = 1 << (loopTime - i - 1);
                dim3 fullBlocksPerGrid((workload + blockSize - 1) / blockSize);
                kernEfficientScanDownSweep CUDA_KERNEL(fullBlocksPerGrid, blockSize) (workload, 1 << i, tmpIn);
            }
        }


        __global__ void kernIsNotZero(int N, int* data, const int* dataIn)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= N)
            {
                return;
            }
            data[index] = (dataIn[index] != 0 ? 1 : 0);
        }

        __global__ void kernScatter(int N, int* odata, const int* idata, const int* NotZeroScan)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= N)
            {
                return;
            }
            if (idata[index] != 0)
			{
				odata[NotZeroScan[index]] = idata[index];
			}
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

            int loopTime = ilog2ceil(n);
            int N = (1 << loopTime);
            int* tmpIn, *tmpInNotZero, * tmpOut;
            cudaMalloc((void**)&tmpIn, sizeof(int) * N);
            cudaMalloc((void**)&tmpInNotZero, sizeof(int) * N);
            cudaMalloc((void**)&tmpOut, sizeof(int) * N);
            cudaMemcpy(tmpIn, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            const int blockSize = 128;
            dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);
            kernIsNotZero CUDA_KERNEL(fullBlocksPerGrid, blockSize) (N, tmpInNotZero, tmpIn);
            scan_impl(loopTime, tmpInNotZero);
            kernScatter CUDA_KERNEL(fullBlocksPerGrid, blockSize) (N, tmpOut, tmpIn, tmpInNotZero);

            timer().endGpuTimer();

            int notZeroCount;
            cudaMemcpy(&notZeroCount, tmpInNotZero + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
            notZeroCount += idata[N - 1] != 0 ? 1 : 0;
            cudaMemcpy(odata, tmpOut, sizeof(int) * notZeroCount, cudaMemcpyDeviceToHost);

            cudaFree(tmpIn);
            cudaFree(tmpInNotZero);
            cudaFree(tmpOut);
            return notZeroCount;
        }
    }
}
