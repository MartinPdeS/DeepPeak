# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/MartinPdeS/DeepPeak/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                 |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| DeepPeak/algorithms/base.py                          |       86 |       71 |       20 |        0 |     14% |28-32, 51-56, 74-134, 147-166, 190-195 |
| DeepPeak/algorithms/cholesky\_solver.py              |       30 |       24 |        6 |        0 |     17% |22-28, 46-69 |
| DeepPeak/algorithms/closed\_form\_solver.py          |      184 |      160 |       30 |        0 |     11% |48-60, 85-92, 103-128, 140, 153-157, 175-197, 220-233, 256-271, 286-308, 321-337, 360-457 |
| DeepPeak/algorithms/non\_maximum\_suppression.py     |      308 |      234 |       58 |        0 |     20% |55, 59, 63, 74, 100-147, 195, 199, 203, 208, 212-213, 229, 269-331, 343-350, 390-403, 421-479, 497, 527-531, 551, 579-594, 612-614, 642-653, 673-751, 762-772, 783-791, 800-802 |
| DeepPeak/algorithms/zero\_crossing.py                |      225 |      169 |       54 |        0 |     20% |51, 54, 68-108, 146, 150, 154, 159, 162-163, 189-226, 271-276, 287-335, 360-442, 458-461, 472-476, 487-497 |
| DeepPeak/dataset.py                                  |       34 |       22 |       14 |        0 |     29% |27-29, 33-82 |
| DeepPeak/helper.py                                   |       18 |       11 |        4 |        0 |     32% |     33-48 |
| DeepPeak/kernel.py                                   |      160 |       12 |       24 |        7 |     89% |33, 35, 151, 213, 250, 321, 353, 408-410, 415, 431 |
| DeepPeak/machine\_learning/classifier/autoencoder.py |       43 |        1 |        4 |        2 |     94% |84->exit, 93 |
| DeepPeak/machine\_learning/classifier/base.py        |       80 |       28 |       18 |        5 |     60% |27-45, 49-50, 54-55, 59, 63-69, 107, 124-126, 135-136, 205, 211->215, 216-217 |
| DeepPeak/machine\_learning/classifier/dense.py       |       32 |        1 |        6 |        2 |     92% |62->exit, 71 |
| DeepPeak/machine\_learning/classifier/metrics.py     |       48 |       12 |        6 |        0 |     67% |18-34, 104 |
| DeepPeak/machine\_learning/classifier/utils.py       |       38 |       38 |        2 |        0 |      0% |     1-245 |
| DeepPeak/machine\_learning/classifier/wavenet.py     |       40 |        1 |        6 |        2 |     93% |66->exit, 77 |
| DeepPeak/signals.py                                  |       57 |        3 |       14 |        5 |     89% |83, 97->101, 127, 175, 183->190 |
| DeepPeak/utils.py                                    |       71 |       71 |       14 |        0 |      0% |     1-185 |
| DeepPeak/visualization.py                            |      202 |      202 |       70 |        0 |      0% |     1-800 |
|                                            **TOTAL** | **1656** | **1060** |  **350** |   **23** | **32%** |           |


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