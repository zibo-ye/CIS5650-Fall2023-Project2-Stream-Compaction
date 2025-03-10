#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            scan_impl(n, odata, idata);
            timer().endCpuTimer();
        }

        void scan_impl(int n, int* odata, const int* idata) {
            int sum = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = sum;
                sum += idata[i];
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int idx = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[idx] = idata[i];
                    idx++;
                }
            }
            timer().endCpuTimer();
            return idx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* is_not_zero = new int[n];
            int* scan_is_not_zero = new int[n];
            timer().startCpuTimer();

            for (int i = 0; i < n; i++) {
                is_not_zero[i] = (idata[i] != 0 ? 1 : 0);
            }
            scan_impl(n, scan_is_not_zero, is_not_zero);

            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
					odata[scan_is_not_zero[i]] = idata[i];
				}
            }

            int count = scan_is_not_zero[n - 1];
            timer().endCpuTimer();
            delete[] is_not_zero;
            delete[] scan_is_not_zero;
            return count;
        }
    }
}
