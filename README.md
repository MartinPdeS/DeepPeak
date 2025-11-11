# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/MartinPdeS/DeepPeak/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                  |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------ | -------: | -------: | -------: | -------: | ---------: | --------: |
| DeepPeak/algorithms/base.py                           |       86 |       71 |       20 |        0 |     14.15% |28-32, 51-56, 74-134, 147-166, 190-195 |
| DeepPeak/algorithms/cholesky\_solver.py               |       30 |       24 |        6 |        0 |     16.67% |22-28, 46-69 |
| DeepPeak/algorithms/closed\_form\_solver.py           |      181 |      160 |       30 |        0 |      9.95% |48-60, 85-92, 103-128, 140, 153-157, 175-197, 220-233, 256-271, 286-308, 321-337, 360-457 |
| DeepPeak/algorithms/non\_maximum\_suppression copy.py |      315 |      315 |       66 |        0 |      0.00% |     1-835 |
| DeepPeak/algorithms/non\_maximum\_suppression.py      |      315 |      256 |       66 |        0 |     15.49% |55, 59, 63, 74, 100-147, 195, 199, 203, 208, 212-213, 229, 253-277, 281-282, 286-288, 292-311, 315-317, 321-322, 339-364, 376-383, 423-436, 454-512, 530, 560-564, 584, 612-627, 645-647, 675-686, 706-784, 795-805, 816-824, 833-835 |
| DeepPeak/algorithms/zero\_crossing.py                 |      205 |      169 |       54 |        0 |     13.90% |51, 54, 68-108, 146, 150, 154, 159, 162-163, 189-226, 271-276, 287-335, 360-442, 458-461, 472-476, 487-497 |
| DeepPeak/dataset.py                                   |       53 |       39 |       20 |        0 |     21.92% |28-30, 36-47, 69-127 |
| DeepPeak/helper.py                                    |       18 |       18 |        4 |        0 |      0.00% |      2-50 |
| DeepPeak/kernel.py                                    |      162 |       16 |       24 |        7 |     86.56% |33, 35, 191, 254, 321, 396, 463, 522-526, 531, 549, 579-583 |
| DeepPeak/machine\_learning/classifier/autoencoder.py  |       42 |        1 |        4 |        2 |     93.48% |84->exit, 93 |
| DeepPeak/machine\_learning/classifier/base.py         |       82 |       22 |       26 |        9 |     65.74% |18-19, 23-24, 28, 32-38, 76, 95-99, 108-111, 155, 160->164, 167->179, 182->186, 209, 217->221, 222-225 |
| DeepPeak/machine\_learning/classifier/dense.py        |       31 |        1 |        6 |        2 |     91.89% |62->exit, 71 |
| DeepPeak/machine\_learning/classifier/metrics.py      |       48 |       12 |        6 |        0 |     66.67% |18-34, 104 |
| DeepPeak/machine\_learning/classifier/utils.py        |       65 |       54 |        6 |        0 |     15.49% |31-51, 112-135, 185-190, 243-259, 298-362 |
| DeepPeak/machine\_learning/classifier/wavenet.py      |      108 |       64 |       32 |        2 |     34.29% |71->exit, 82, 132-189, 210-276 |
| DeepPeak/signals.py                                   |       60 |        3 |       14 |        5 |     89.19% |83, 101->105, 136, 186, 194->201 |
| DeepPeak/utils.py                                     |       71 |       71 |       14 |        0 |      0.00% |     1-185 |
| DeepPeak/visualization.py                             |      202 |      202 |       70 |        0 |      0.00% |     1-800 |
|                                             **TOTAL** | **2074** | **1498** |  **468** |   **27** | **24.51%** |           |


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