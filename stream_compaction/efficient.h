#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

		void scan(int n, int* odata, const int* idata);
		void scan_impl(int loopTime, int* tmpIn);

        int compact(int n, int *odata, const int *idata);
    }
}
