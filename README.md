# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/MartinPdeS/DeepPeak/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                 |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|----------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| DeepPeak/algorithms/base.py                          |       86 |       66 |       20 |        0 |     18.87% |28-32, 51-56, 74-134, 147-166 |
| DeepPeak/algorithms/cholesky\_solver.py              |       30 |       24 |        6 |        0 |     16.67% |22-28, 46-69 |
| DeepPeak/algorithms/closed\_form\_solver.py          |      181 |      160 |       30 |        0 |      9.95% |48-60, 85-92, 103-128, 140, 153-157, 175-197, 220-233, 256-271, 286-308, 321-337, 360-457 |
| DeepPeak/algorithms/non\_maximum\_suppression.py     |      315 |      256 |       66 |        0 |     15.49% |55, 59, 63, 74, 100-147, 195, 199, 203, 208, 212-213, 229, 253-277, 281-282, 286-288, 292-311, 315-317, 321-322, 339-364, 376-383, 423-436, 454-512, 530, 560-564, 584, 612-627, 645-647, 675-686, 706-784, 795-805, 816-824, 833-835 |
| DeepPeak/algorithms/peak\_locator.py                 |       72 |        9 |       30 |        7 |     82.35% |34, 49, 51, 53, 56, 60-63, 85 |
| DeepPeak/algorithms/zero\_crossing.py                |      205 |      169 |       54 |        0 |     13.90% |51, 54, 68-108, 146, 150, 154, 159, 162-163, 189-226, 271-276, 287-335, 360-442, 458-461, 472-476, 487-497 |
| DeepPeak/analysis/dead\_time.py                      |       70 |       55 |       30 |        3 |     18.00% |46-47, 84-106, 127-128, 143, 145, 147, 206-262 |
| DeepPeak/analysis/dilution\_series.py                |      550 |      116 |      154 |       54 |     72.44% |59, 74, 81-83, 334-340, 351, 356, 365, 388-391, 394-395, 410, 424-\>426, 426-\>428, 429, 431, 433, 478-\>483, 578-580, 611-627, 631-\>633, 634, 636, 638, 676-677, 682-686, 706-713, 732, 734, 736, 750-751, 756-760, 791, 793, 795, 814-815, 880, 882, 884, 924, 933, 940-\>944, 949-1005, 1068-1074, 1129-1134, 1145, 1166, 1168, 1170, 1201, 1210, 1254, 1261-1269, 1330, 1429-1452, 1470, 1635, 1658-\>1660, 1661 |
| DeepPeak/analysis/distributions/amplitude.py         |       66 |       14 |       22 |        4 |     75.00% |24-35, 45, 61-\>64, 119, 151-157 |
| DeepPeak/analysis/distributions/event\_arrival.py    |      102 |       18 |       32 |       11 |     75.37% |64-\>68, 74-83, 107-110, 119-120, 123, 134, 143, 154, 165, 172-176, 210 |
| DeepPeak/analysis/distributions/width.py             |       71 |       20 |       24 |        6 |     66.32% |19-30, 40, 48-\>51, 52-65, 73, 96, 125-131 |
| DeepPeak/analysis/metrics/detection.py               |       13 |        2 |        0 |        0 |     84.62% |    24, 30 |
| DeepPeak/analysis/metrics/distributions.py           |      353 |       78 |      182 |       75 |     69.16% |82-114, 139-140, 145-\>199, 153, 170-\>182, 184-\>197, 202, 204-\>212, 212-\>214, 215, 217, 219, 237-238, 242-\>271, 276, 278-\>286, 286-\>288, 289, 291, 293, 314-315, 320-\>346, 349, 351-\>359, 359-\>361, 362, 364, 366, 420-421, 428, 442-\>461, 466, 468-\>476, 476-\>478, 479, 481, 483, 507-\>540, 518, 543, 545-\>553, 556, 558, 560, 577-578, 595-\>614, 618, 620-\>628, 628-\>630, 631, 633, 635, 700, 714-\>737, 742, 744-\>752, 755, 757, 759, 777-778, 783-\>818, 798, 823, 825-\>833, 833-\>835, 836, 838, 840, 857-858, 875-\>898, 902, 904-\>912, 912-\>914, 915, 917, 919 |
| DeepPeak/analysis/metrics/trace\_record.py           |      407 |       43 |      160 |       44 |     84.66% |32, 38, 122, 166-\>185, 195, 197, 210, 212, 214, 295, 297, 336-\>370, 362-\>370, 443, 468, 470, 483, 485, 487, 654-655, 661, 672-674, 719, 729, 738, 746, 794-\>799, 799-\>802, 816, 818, 820, 1005-1006, 1013, 1015, 1028-\>1039, 1036-\>1039, 1040-1042, 1094, 1104, 1113, 1121, 1192-\>1197, 1197-\>1200, 1214, 1216, 1218 |
| DeepPeak/analysis/metrics/utils.py                   |        7 |        1 |        4 |        1 |     81.82% |        15 |
| DeepPeak/analysis/trace\_io.py                       |       83 |       59 |       10 |        0 |     25.81% |37-41, 53, 65, 77, 88-93, 113-135, 146-151, 167-170, 186-190, 206-209, 233-242, 254, 273-299 |
| DeepPeak/analysis/trace\_plots.py                    |      157 |      139 |       70 |        2 |      8.81% |30, 58-71, 77-356 |
| DeepPeak/analysis/triggers.py                        |       46 |        6 |       14 |        6 |     80.00% |21, 24, 27, 30, 57, 79 |
| DeepPeak/analysis/wavenet\_trace.py                  |      256 |       42 |       88 |       26 |     77.33% |35, 45-56, 74, 79, 96, 111-116, 120, 136, 148, 150, 162, 170-171, 195, 204, 208, 220-221, 291-297, 330-\>337, 334-\>337, 499-\>510, 638, 647, 656, 664-668, 679, 702, 709-710 |
| DeepPeak/dataset.py                                  |      282 |      132 |      134 |       24 |     46.63% |46-\>52, 58, 60-\>63, 63-\>exit, 68-70, 79, 103-107, 143, 181-250, 282-333, 365, 375-\>379, 416, 422, 428, 433, 437, 442, 481, 485-491, 522-536, 616, 630, 642-646, 656, 661, 673-680, 711-716, 719-729 |
| DeepPeak/kernels/base.py                             |      154 |       39 |       62 |       13 |     68.52% |41, 58, 60, 94, 108-130, 133, 177-\>181, 190-\>193, 233-\>237, 240-\>243, 244, 280-288, 302, 304, 313-323 |
| DeepPeak/kernels/custom.py                           |       41 |       25 |       10 |        2 |     35.29% |39, 43-46, 49, 88-102, 132-171 |
| DeepPeak/kernels/dirac.py                            |       40 |       25 |        6 |        0 |     32.61% |76-115, 148-152 |
| DeepPeak/kernels/lorentzian.py                       |       17 |        5 |        0 |        0 |     70.59% |38, 70-82, 107 |
| DeepPeak/kernels/square.py                           |       21 |        8 |        0 |        0 |     61.90% |38, 76-88, 122-126 |
| DeepPeak/kernels/two\_lobe\_gaussian.py              |       57 |        2 |        2 |        1 |     94.92% |   75, 107 |
| DeepPeak/machine\_learning/classifier/autoencoder.py |       42 |        1 |        4 |        2 |     93.48% |84-\>exit, 93 |
| DeepPeak/machine\_learning/classifier/base.py        |      135 |       32 |       60 |       19 |     69.74% |21-22, 26-27, 31, 35-41, 79, 98-102, 111-114, 390, 392, 397-398, 401, 409, 419-\>416, 423, 428, 435, 443-445, 451-452, 468-\>467, 477-\>482, 482-\>474, 490-\>exit, 494-\>497, 497-\>500 |
| DeepPeak/machine\_learning/classifier/dense.py       |       31 |        1 |        6 |        2 |     91.89% |62-\>exit, 71 |
| DeepPeak/machine\_learning/classifier/losses.py      |       79 |        2 |        8 |        2 |     95.40% |  197, 231 |
| DeepPeak/machine\_learning/classifier/metrics.py     |       48 |       12 |        6 |        0 |     66.67% |18-34, 104 |
| DeepPeak/machine\_learning/classifier/utils.py       |       65 |       65 |        6 |        0 |      0.00% |     1-362 |
| DeepPeak/machine\_learning/classifier/wavenet.py     |      144 |       27 |       36 |        9 |     77.78% |70, 81-92, 101-103, 110-123, 133, 140-141, 217, 339-355, 363, 396-\>401, 401-\>406 |
| DeepPeak/peak\_count.py                              |       64 |        7 |       14 |        6 |     83.33% |29, 39, 41, 50, 54, 61, 98 |
| DeepPeak/processing.py                               |      168 |      106 |       70 |       15 |     33.19% |8, 10, 17, 25, 27, 36, 53-72, 85, 157-170, 181-183, 187-192, 203-205, 212-219, 274-343, 380, 383-384, 387-388, 400-410 |
| DeepPeak/signal\_generator.py                        |      139 |       87 |       62 |        4 |     31.84% |40-42, 47-52, 57, 97, 130-134, 137-138, 141-145, 159-204, 212, 224-296, 300-310 |
| DeepPeak/utils/datasets.py                           |       16 |       13 |        4 |        0 |     15.00% |     12-37 |
| DeepPeak/utils/deconvolution.py                      |       54 |       47 |       18 |        0 |      9.72% |12-19, 22-23, 28-56, 59-122 |
| DeepPeak/utils/history.py                            |       28 |       23 |       12 |        0 |     12.50% |8-14, 24-47 |
| DeepPeak/utils/io.py                                 |       16 |       13 |        6 |        0 |     13.64% |     40-59 |
| DeepPeak/utils/iterables.py                          |       10 |        1 |        4 |        1 |     85.71% |        15 |
| DeepPeak/utils/signal\_processing.py                 |       58 |       23 |       14 |        5 |     55.56% |17-45, 51, 65, 90-91, 94-95, 102 |
| DeepPeak/visualization.py                            |      152 |      152 |       62 |        0 |      0.00% |     1-602 |
| **TOTAL**                                            | **4963** | **2125** | **1602** |  **344** | **52.55%** |           |

4 files skipped due to complete coverage.


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