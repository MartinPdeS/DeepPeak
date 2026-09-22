# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/MartinPdeS/DeepPeak/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                               |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|--------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| DeepPeak/analysis/comparison.py                    |      146 |       20 |       40 |       12 |     80.65% |39-41, 51-\>50, 53, 82, 89, 106, 109-113, 119, 131, 157, 164, 169, 180, 193, 287, 322-\>329, 329-\>339 |
| DeepPeak/analysis/dead\_time.py                    |       69 |       55 |       30 |        3 |     17.17% |45-46, 83-105, 126-127, 142, 144, 146, 205-261 |
| DeepPeak/analysis/dilution\_series.py              |      526 |      115 |      140 |       48 |     71.62% |68, 83, 90-92, 343-349, 360, 365, 376, 399-402, 407-408, 423, 437-\>439, 439-\>441, 442, 444, 446, 491-\>496, 591-593, 624-640, 646-\>648, 649, 651, 653, 691-692, 697-701, 723-730, 749, 751, 753, 767-768, 773-777, 810, 812, 814, 835-836, 903, 905, 907, 947, 956, 963-\>967, 972-1030, 1055-1062, 1064, 1119-1125, 1254, 1352-1387, 1405, 1495-\>1497, 1525, 1566-\>1568, 1569 |
| DeepPeak/analysis/distributions/amplitude.py       |       66 |       14 |       22 |        4 |     75.00% |24-35, 45, 61-\>64, 119, 151-157 |
| DeepPeak/analysis/distributions/event\_arrival.py  |      102 |       18 |       32 |       11 |     75.37% |64-\>68, 74-83, 107-110, 119-120, 123, 134, 143, 154, 165, 172-176, 210 |
| DeepPeak/analysis/distributions/width.py           |       71 |       20 |       24 |        6 |     66.32% |19-30, 40, 48-\>51, 52-65, 73, 96, 125-131 |
| DeepPeak/analysis/metrics/distributions.py         |      374 |       85 |      186 |       77 |     68.93% |18-24, 35-37, 109-141, 166-167, 172-\>226, 180, 197-\>209, 211-\>224, 229, 231-\>239, 239-\>241, 242, 244, 246, 264-265, 269-\>298, 303, 305-\>313, 313-\>315, 316, 318, 320, 341-342, 347-\>373, 376, 378-\>386, 386-\>388, 389, 391, 393, 448-449, 456, 470-\>506, 482, 511, 513-\>521, 521-\>523, 524, 526, 528, 552-\>585, 563, 588, 590-\>598, 601, 603, 605, 622-623, 640-\>659, 663, 665-\>673, 673-\>675, 676, 678, 680, 746, 760-\>798, 776, 803, 805-\>813, 816, 818, 820, 838-839, 844-\>879, 859, 884, 886-\>894, 894-\>896, 897, 899, 901, 918-919, 936-\>959, 963, 965-\>973, 973-\>975, 976, 978, 980 |
| DeepPeak/analysis/metrics/trace\_record.py         |      504 |       65 |      182 |       46 |     82.65% |26-33, 53, 59, 177, 183, 190, 260, 262, 275, 277, 432-433, 440, 442, 486-\>520, 512-\>520, 616, 618, 630-703, 706-709, 719, 721, 723, 892-893, 899, 910-911, 956, 966, 975, 983, 1032-\>1037, 1037-\>1040, 1054, 1056, 1058, 1245-1246, 1253, 1255, 1268-\>1279, 1276-\>1279, 1280-1281, 1343, 1352, 1360, 1432-\>1437, 1437-\>1440, 1454, 1456, 1458 |
| DeepPeak/analysis/metrics/utils.py                 |        7 |        1 |        4 |        1 |     81.82% |        15 |
| DeepPeak/analysis/noise\_analysis.py               |      217 |       29 |       60 |       21 |     81.23% |28, 30, 34, 39, 44, 50, 56, 76-80, 94-95, 111, 174, 176, 184-185, 190, 212, 217, 269-\>280, 271-\>273, 301-\>315, 304-\>315, 352, 367-373, 387 |
| DeepPeak/analysis/pulse\_shape.py                  |      432 |      101 |      160 |       59 |     69.59% |33, 37, 42, 46, 54, 74-81, 88, 95, 107, 111, 125, 128, 132, 139-148, 177, 182, 184, 193-195, 205-206, 210, 213-216, 243-244, 258, 264, 302, 334-341, 361, 363, 364-\>366, 367, 386, 390-\>392, 394, 409, 413, 418, 421, 460, 468, 476, 482, 516, 542-\>551, 551-\>571, 555-\>571, 561-\>571, 586-\>593, 639, 644, 649, 655-\>658, 659, 748, 762, 766-769, 774, 779, 788-793, 796, 799-813, 821-827, 844, 850, 865, 872 |
| DeepPeak/analysis/series\_calculations.py          |       41 |        4 |       16 |        4 |     85.96% |34, 42, 79, 88 |
| DeepPeak/analysis/wavenet\_trace.py                |      363 |       96 |      136 |       37 |     67.74% |43, 55-66, 84, 89, 106, 117, 132-137, 141, 157, 169, 171, 185, 193-194, 218, 227, 231, 251-252, 274, 327-333, 366-\>373, 370-\>373, 403, 436-445, 504-514, 516, 545-554, 578-580, 584-586, 625-633, 644-\>656, 702-719, 756-766, 810-811, 883-894, 900, 909, 918, 926-930, 941, 964, 971-972 |
| DeepPeak/benchmarking.py                           |      137 |        1 |       40 |        3 |     97.74% |153, 176-\>174, 212-\>202 |
| DeepPeak/core/config.py                            |      181 |       35 |       76 |       29 |     73.54% |16, 18, 20, 77, 96-102, 135, 169, 173, 177-178, 184, 213, 215, 219, 250, 252, 300, 302, 306, 310, 312, 363, 365, 384, 386, 393, 429, 433, 462, 464 |
| DeepPeak/core/types.py                             |      103 |       12 |       28 |        9 |     83.97% |17, 27, 31, 33, 62, 114, 152, 159, 169, 239, 301, 312 |
| DeepPeak/detection/base.py                         |       86 |        8 |       20 |        8 |     84.91% |92-93, 106, 116, 127, 137, 149, 170, 180-\>183 |
| DeepPeak/detection/cholesky\_solver.py             |       30 |        1 |        6 |        1 |     94.44% |        59 |
| DeepPeak/detection/closed\_form\_solver.py         |      181 |      137 |       30 |        3 |     22.27% |50-74, 120, 126, 129-136, 156, 195-217, 242-255, 280-295, 310-332, 345-361, 386-495 |
| DeepPeak/detection/non\_maximum\_suppression.py    |      315 |      162 |       66 |       11 |     44.09% |111-\>115, 115-\>119, 146, 150-\>154, 154-\>157, 212, 216, 220, 225, 229-230, 246, 270-296, 302-303, 311-313, 317-338, 344-346, 356-357, 374-401, 413-420, 530-531, 551-553, 678-683, 692-694, 714-716, 747, 749, 779-861, 872-882, 893-901, 910-912 |
| DeepPeak/detection/peak\_locator.py                |      154 |       29 |       74 |       18 |     75.88% |10, 46, 55-57, 141, 145-148, 152, 159, 164-171, 174, 184-\>188, 192-\>196, 235, 250, 252, 254, 257, 261-264, 286 |
| DeepPeak/detection/triggers.py                     |       63 |       17 |       20 |        6 |     65.06% |21, 24, 27, 30, 57, 79, 125-131, 134-137 |
| DeepPeak/detection/zero\_crossing.py               |      206 |      170 |       54 |        0 |     13.85% |51, 54, 70-112, 150, 154, 158, 163, 166-167, 193-249, 294-301, 314-365, 392-478, 494-497, 508-514, 525-535 |
| DeepPeak/generation/dataset.py                     |      325 |      140 |      176 |       35 |     51.90% |98-\>104, 110, 112-\>115, 115-\>119, 134, 145, 153, 165, 210, 283, 294, 298, 301, 347, 349-356, 359-\>361, 384-386, 415, 439-443, 501, 535-591, 623-674, 709, 715, 721, 726, 730, 735, 774, 778-784, 815-829, 909, 923, 935-939, 949, 954, 966-973, 1004-1009, 1014, 1017-1022 |
| DeepPeak/generation/kernels/base.py                |      175 |       37 |       74 |       16 |     73.09% |35-\>40, 38-39, 50, 67, 69, 134, 152-174, 177, 182-\>184, 184-\>186, 186-\>188, 232, 262-\>282, 264-\>282, 329, 365-373, 386, 388 |
| DeepPeak/generation/kernels/custom.py              |      180 |       44 |       42 |        5 |     68.92% |104-142, 155, 181, 317-318, 412-426, 439, 441, 443 |
| DeepPeak/generation/kernels/dirac.py               |       39 |       26 |        6 |        0 |     28.89% |68-107, 140-144 |
| DeepPeak/generation/kernels/lorentzian.py          |       18 |        6 |        0 |        0 |     66.67% |38, 69-81, 106 |
| DeepPeak/generation/kernels/square.py              |       21 |        9 |        0 |        0 |     57.14% |37, 69-81, 115-119 |
| DeepPeak/generation/kernels/two\_lobe\_gaussian.py |       76 |        2 |        8 |        4 |     92.86% |82, 93-\>95, 95-\>97, 97-\>99, 124 |
| DeepPeak/generation/noises/base.py                 |       34 |        6 |       14 |        5 |     77.08% |18, 35, 37, 42, 44-45 |
| DeepPeak/generation/noises/correlated\_gaussian.py |       28 |       18 |        6 |        0 |     29.41% |33-38, 64-80 |
| DeepPeak/generation/peak\_count.py                 |       67 |        7 |       14 |        6 |     83.95% |31, 41, 43, 56, 60, 67, 109 |
| DeepPeak/generation/signal\_generator.py           |      198 |       39 |      102 |       21 |     76.00% |81, 90, 156, 158, 162, 165, 183-187, 237-238, 241-245, 285, 315-\>296, 345, 360-368, 380-387, 391, 401, 423, 425-426, 448, 450, 462-464, 466 |
| DeepPeak/io/trace\_io.py                           |       83 |        1 |       10 |        3 |     95.70% |91, 278-\>286, 286-\>292 |
| DeepPeak/models/base.py                            |      192 |      124 |       92 |       14 |     29.58% |29-30, 44-47, 51, 55-61, 109-\>111, 142-149, 192-195, 239-\>248, 244, 265, 269-\>271, 272, 282, 284, 291, 293, 311-314, 343-352, 354, 356, 611-722 |
| DeepPeak/models/dense.py                           |       30 |        2 |        6 |        2 |     88.89% |    64, 72 |
| DeepPeak/models/evaluation.py                      |       24 |        3 |        4 |        2 |     82.14% |66, 78, 106 |
| DeepPeak/models/losses.py                          |       95 |        3 |       12 |        4 |     93.46% |11-\>13, 197, 231, 287 |
| DeepPeak/models/plotting.py                        |       60 |        8 |       34 |       13 |     77.66% |64, 72, 74, 86-\>88, 97, 100-\>110, 101-\>110, 102-\>101, 123-\>128, 139, 146, 149, 152 |
| DeepPeak/models/training.py                        |       59 |        7 |       26 |        7 |     83.53% |68, 70, 72, 74, 76, 78, 105-\>122, 166 |
| DeepPeak/models/unet1d.py                          |      106 |       47 |       18 |        2 |     52.42% |56, 86, 134-158, 167-247 |
| DeepPeak/models/wavenet.py                         |      146 |       29 |       40 |       11 |     76.34% |70, 81-92, 101-103, 110-123, 133, 140-141, 211, 223, 248, 353-379, 387, 425-\>430, 430-\>435 |
| DeepPeak/pipeline.py                               |       74 |       17 |       26 |        7 |     68.00% |105, 141-143, 154-163, 166, 169, 215, 223, 284-290 |
| DeepPeak/plotting/trace\_plots.py                  |      167 |      143 |       76 |        3 |     12.76% |30, 60-62, 67-69, 77-367 |
| DeepPeak/plotting/trace\_record.py                 |       20 |        9 |        8 |        1 |     42.86% |14-\>16, 24-26, 34-36, 44-46 |
| DeepPeak/processing.py                             |      170 |       12 |       74 |       13 |     89.75% |54, 66-\>72, 165, 189-\>194, 203-205, 213-\>217, 219, 285, 296, 313-\>318, 327-329, 337-\>341, 342 |
| DeepPeak/utils/datasets.py                         |       16 |       13 |        4 |        0 |     15.00% |     12-37 |
| DeepPeak/utils/history.py                          |       28 |       23 |       12 |        0 |     12.50% |8-14, 24-47 |
| DeepPeak/utils/io.py                               |       16 |       13 |        6 |        0 |     13.64% |     40-59 |
| DeepPeak/utils/iterables.py                        |       10 |        1 |        4 |        1 |     85.71% |        15 |
| DeepPeak/utils/signal\_processing.py               |       63 |       23 |       18 |        5 |     60.49% |15-43, 49, 100, 133-134, 137-138, 145 |
| **TOTAL**                                          | **7033** | **2007** | **2360** |  **597** | **66.10%** |           |

10 files skipped due to complete coverage.


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