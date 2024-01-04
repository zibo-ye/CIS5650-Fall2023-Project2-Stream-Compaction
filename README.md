CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* My name: Zibo Ye
  * [LinkedIn](https://www.linkedin.com/in/zibo-ye/)
  * [CMU ETC Intro Page](https://www.etc.cmu.edu/blog/author/ziboy/)
  * [Twitter](https://twitter.com/zibo_ye)
  * This is a guy trying to self-learn ways to optimize GPU codes.
* Tested on: Windows 11, AMD R7-5800H @ 3.2GHz (Up to 4.4Ghz) 32GB, RTX 3080 Mobile 16GB

### Result

```
****************
** SCAN TESTS **
****************
    [  42  19  25  44  14   4  30  32  46  48  49  15   8 ...   9   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 9.4504ms    (std::chrono Measured)
    [   0  42  61  86 130 144 148 178 210 256 304 353 368 ... 411048393 411048402 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 10.1574ms    (std::chrono Measured)
    [   0  42  61  86 130 144 148 178 210 256 304 353 368 ... 411048286 411048332 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 10.5875ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 9.82234ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 3.98925ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 3.8584ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.22739ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.1313ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   3   3   0   2   0   0   0   0   2   3   3   0 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 25.9578ms    (std::chrono Measured)
    [   3   3   2   2   3   3   2   1   3   1   2   2   3 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 25.7097ms    (std::chrono Measured)
    [   3   3   2   2   3   3   2   1   3   1   2   2   3 ...   3   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 63.379ms    (std::chrono Measured)
    [   3   3   2   2   3   3   2   1   3   1   2   2   3 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 5.25258ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 4.56541ms    (CUDA Measured)
    passed
```