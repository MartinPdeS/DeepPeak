# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/MartinPdeS/DeepPeak/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                               |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|--------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| DeepPeak/analysis/comparison.py                    |      150 |       20 |       40 |       12 |     81.05% |42-44, 54-\>53, 56, 85, 92, 109, 112-116, 122, 134, 160, 167, 172, 183, 196, 290, 325-\>332, 332-\>342 |
| DeepPeak/analysis/dead\_time.py                    |       70 |       55 |       30 |        3 |     18.00% |46-47, 84-106, 127-128, 143, 145, 147, 206-262 |
| DeepPeak/analysis/dilution\_series.py              |      536 |      115 |      146 |       48 |     72.29% |68, 83, 90-92, 343-349, 360, 365, 376, 399-402, 407-408, 423, 437-\>439, 439-\>441, 442, 444, 446, 491-\>496, 591-593, 624-640, 646-\>648, 649, 651, 653, 691-692, 697-701, 723-730, 749, 751, 753, 767-768, 773-777, 810, 812, 814, 835-836, 903, 905, 907, 947, 956, 963-\>967, 972-1030, 1055-1062, 1064, 1119-1125, 1254, 1352-1387, 1405, 1495-\>1497, 1525, 1566-\>1568, 1569 |
| DeepPeak/analysis/distributions/amplitude.py       |       66 |       14 |       22 |        4 |     75.00% |24-35, 45, 61-\>64, 119, 151-157 |
| DeepPeak/analysis/distributions/event\_arrival.py  |      102 |       18 |       32 |       11 |     75.37% |64-\>68, 74-83, 107-110, 119-120, 123, 134, 143, 154, 165, 172-176, 210 |
| DeepPeak/analysis/distributions/width.py           |       71 |       20 |       24 |        6 |     66.32% |19-30, 40, 48-\>51, 52-65, 73, 96, 125-131 |
| DeepPeak/analysis/metrics/distributions.py         |      374 |       85 |      186 |       77 |     68.93% |18-24, 35-37, 109-141, 166-167, 172-\>226, 180, 197-\>209, 211-\>224, 229, 231-\>239, 239-\>241, 242, 244, 246, 264-265, 269-\>298, 303, 305-\>313, 313-\>315, 316, 318, 320, 341-342, 347-\>373, 376, 378-\>386, 386-\>388, 389, 391, 393, 448-449, 456, 470-\>506, 482, 511, 513-\>521, 521-\>523, 524, 526, 528, 552-\>585, 563, 588, 590-\>598, 601, 603, 605, 622-623, 640-\>659, 663, 665-\>673, 673-\>675, 676, 678, 680, 746, 760-\>798, 776, 803, 805-\>813, 816, 818, 820, 838-839, 844-\>879, 859, 884, 886-\>894, 894-\>896, 897, 899, 901, 918-919, 936-\>959, 963, 965-\>973, 973-\>975, 976, 978, 980 |
| DeepPeak/analysis/metrics/trace\_record.py         |      510 |       67 |      182 |       46 |     82.51% |26-33, 53, 59, 182, 188, 195, 265, 267, 280, 282, 437-438, 445, 447, 491-\>525, 517-\>525, 621, 623, 635-708, 711-714, 724, 726, 728, 897-898, 904, 915-917, 963, 973, 982, 990, 1039-\>1044, 1044-\>1047, 1061, 1063, 1065, 1252-1253, 1260, 1262, 1275-\>1286, 1283-\>1286, 1287-1289, 1352, 1361, 1369, 1441-\>1446, 1446-\>1449, 1463, 1465, 1467 |
| DeepPeak/analysis/metrics/utils.py                 |        7 |        1 |        4 |        1 |     81.82% |        15 |
| DeepPeak/analysis/noise\_analysis.py               |      217 |       29 |       60 |       21 |     81.23% |28, 30, 34, 39, 44, 50, 56, 76-80, 94-95, 111, 174, 176, 184-185, 190, 212, 217, 269-\>280, 271-\>273, 301-\>315, 304-\>315, 352, 367-373, 387 |
| DeepPeak/analysis/pulse\_shape.py                  |      432 |      101 |      160 |       59 |     69.59% |33, 37, 42, 46, 54, 74-81, 88, 95, 107, 111, 125, 128, 132, 139-148, 177, 182, 184, 193-195, 205-206, 210, 213-216, 243-244, 258, 264, 302, 334-341, 361, 363, 364-\>366, 367, 386, 390-\>392, 394, 409, 413, 418, 421, 460, 468, 476, 482, 516, 542-\>551, 551-\>571, 555-\>571, 561-\>571, 586-\>593, 639, 644, 649, 655-\>658, 659, 748, 762, 766-769, 774, 779, 788-793, 796, 799-813, 821-827, 844, 850, 865, 872 |
| DeepPeak/analysis/series\_calculations.py          |       41 |        4 |       16 |        4 |     85.96% |34, 42, 79, 88 |
| DeepPeak/analysis/wavenet\_trace.py                |      364 |       96 |      136 |       37 |     67.80% |44, 56-67, 85, 90, 107, 118, 133-138, 142, 158, 170, 172, 186, 194-195, 219, 228, 232, 252-253, 275, 328-334, 367-\>374, 371-\>374, 404, 437-446, 505-515, 517, 546-553, 577-579, 583-585, 624-632, 643-\>655, 701-718, 755-765, 813-814, 886-897, 903, 912, 921, 929-933, 944, 967, 974-975 |
| DeepPeak/core/config.py                            |       69 |       11 |       30 |       10 |     76.77% |33, 53, 57, 61-62, 68, 85, 87, 91, 108, 110 |
| DeepPeak/core/types.py                             |      114 |       12 |       28 |        9 |     85.21% |19, 29, 33, 35, 64, 93, 115, 122, 132, 181, 217, 222 |
| DeepPeak/detection/base.py                         |       86 |       71 |       20 |        0 |     14.15% |30-34, 53-58, 78-157, 170-189, 215-220 |
| DeepPeak/detection/cholesky\_solver.py             |       30 |       24 |        6 |        0 |     16.67% |22-28, 48-71 |
| DeepPeak/detection/closed\_form\_solver.py         |      181 |      160 |       30 |        0 |      9.95% |50-74, 99-106, 117-144, 156, 171-175, 195-217, 242-255, 280-295, 310-332, 345-361, 386-495 |
| DeepPeak/detection/non\_maximum\_suppression.py    |      315 |      256 |       66 |        0 |     15.49% |55, 59, 63, 78, 104-164, 212, 216, 220, 225, 229-230, 246, 270-296, 302-303, 311-313, 317-338, 344-346, 356-357, 374-401, 413-420, 468-483, 503-569, 587, 617-625, 647, 677-696, 714-716, 746-757, 779-861, 872-882, 893-901, 910-912 |
| DeepPeak/detection/peak\_locator.py                |      154 |       29 |       74 |       18 |     75.88% |10, 46, 55-57, 141, 145-148, 152, 159, 164-171, 174, 184-\>188, 192-\>196, 235, 250, 252, 254, 257, 261-264, 286 |
| DeepPeak/detection/triggers.py                     |       63 |       17 |       20 |        6 |     65.06% |21, 24, 27, 30, 57, 79, 125-131, 134-137 |
| DeepPeak/detection/zero\_crossing.py               |      205 |      169 |       54 |        0 |     13.90% |51, 54, 70-112, 150, 154, 158, 163, 166-167, 193-249, 294-301, 314-366, 393-479, 495-498, 509-515, 526-536 |
| DeepPeak/generation/dataset.py                     |      257 |      127 |      128 |       22 |     44.68% |46-\>52, 58, 60-\>63, 63-\>exit, 68-70, 79, 103-107, 143, 177-233, 265-316, 351, 357, 363, 368, 372, 377, 416, 420-426, 457-471, 551, 565, 577-581, 591, 596, 608-615, 646-651, 654-664 |
| DeepPeak/generation/kernels/base.py                |      182 |       42 |       80 |       17 |     70.61% |35-\>40, 38-39, 50, 67, 69, 134, 152-174, 177, 182-\>184, 184-\>186, 186-\>188, 232, 262-\>282, 264-\>282, 333, 369-377, 392, 394, 405-415 |
| DeepPeak/generation/kernels/custom.py              |      180 |      100 |       42 |        2 |     39.64% |104-142, 155, 181, 327-328, 422-436, 447-468, 476, 489-533, 546-594 |
| DeepPeak/generation/kernels/dirac.py               |       41 |       26 |        6 |        0 |     31.91% |77-118, 151-155 |
| DeepPeak/generation/kernels/lorentzian.py          |       19 |        6 |        0 |        0 |     68.42% |39, 72-86, 111 |
| DeepPeak/generation/kernels/square.py              |       22 |        9 |        0 |        0 |     59.09% |38, 77-91, 125-129 |
| DeepPeak/generation/kernels/two\_lobe\_gaussian.py |       77 |        2 |        8 |        4 |     92.94% |83, 94-\>96, 96-\>98, 98-\>100, 125 |
| DeepPeak/generation/noises/base.py                 |       35 |        6 |       14 |        5 |     77.55% |20, 37, 39, 44, 46-47 |
| DeepPeak/generation/peak\_count.py                 |       67 |        7 |       14 |        6 |     83.95% |31, 41, 43, 56, 60, 67, 109 |
| DeepPeak/generation/signal\_generator.py           |      185 |       92 |       94 |        9 |     46.59% |43-45, 50-55, 60, 125, 127, 132, 196-197, 200-204, 228-273, 281, 293-365, 369-379, 400, 402, 414-416, 418 |
| DeepPeak/io/trace\_io.py                           |       83 |       59 |       10 |        0 |     25.81% |37-41, 53, 65, 77, 88-93, 113-135, 146-151, 167-170, 186-190, 206-209, 233-242, 254, 273-299 |
| DeepPeak/models/base.py                            |      147 |       39 |       66 |       21 |     67.14% |24-25, 29-30, 34, 38-44, 95-99, 108-111, 140-145, 147, 149, 405, 407, 412-413, 416, 424, 434-\>431, 438, 443, 450, 458-460, 466-467, 483-\>482, 492-\>497, 497-\>489, 505-\>exit, 509-\>512, 512-\>515 |
| DeepPeak/models/dense.py                           |       30 |        2 |        6 |        2 |     88.89% |    64, 72 |
| DeepPeak/models/losses.py                          |       98 |        3 |       12 |        4 |     93.64% |13-\>15, 163, 184, 214 |
| DeepPeak/models/metrics.py                         |       27 |       27 |        2 |        0 |      0.00% |      3-40 |
| DeepPeak/models/plotting.py                        |       14 |       14 |        4 |        0 |      0.00% |      3-29 |
| DeepPeak/models/training.py                        |       40 |       19 |       16 |        0 |     37.50% |38-49, 54-72, 77 |
| DeepPeak/models/unet1d.py                          |      104 |       45 |       16 |        2 |     54.17% |56, 86, 134-158, 167-244 |
| DeepPeak/models/wavenet.py                         |      147 |       29 |       38 |       11 |     76.22% |72, 83-94, 103-105, 112-125, 135, 142-143, 213, 225, 250, 356-378, 386, 421-\>426, 426-\>431 |
| DeepPeak/plotting/trace\_plots.py                  |      168 |      143 |       76 |        3 |     13.11% |31, 61-63, 68-70, 78-368 |
| DeepPeak/plotting/trace\_record.py                 |       20 |        9 |        8 |        1 |     42.86% |19-\>21, 29-31, 39-41, 49-51 |
| DeepPeak/processing.py                             |      168 |      110 |       70 |       15 |     30.67% |8, 10, 17, 25, 27, 36, 53-72, 85, 157-170, 181-183, 187-192, 203-205, 212-219, 274-343, 380, 383-384, 387-388, 395-410 |
| DeepPeak/utils/datasets.py                         |       16 |       13 |        4 |        0 |     15.00% |     12-37 |
| DeepPeak/utils/history.py                          |       28 |       23 |       12 |        0 |     12.50% |8-14, 24-47 |
| DeepPeak/utils/io.py                               |       16 |       13 |        6 |        0 |     13.64% |     40-59 |
| DeepPeak/utils/iterables.py                        |       10 |        1 |        4 |        1 |     85.71% |        15 |
| DeepPeak/utils/signal\_processing.py               |       64 |       23 |       18 |        5 |     60.98% |17-45, 51, 102, 135-136, 139-140, 147 |
| **TOTAL**                                          | **6526** | **2363** | **2120** |  **502** | **58.88%** |           |

9 files skipped due to complete coverage.


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/MartinPdeS/DeepPeak/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/MartinPdeS/DeepPeak/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/MartinPdeS/DeepPeak/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/MartinPdeS/DeepPeak/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FMartinPdeS%2FDeepPeak%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/MartinPdeS/DeepPeak/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.