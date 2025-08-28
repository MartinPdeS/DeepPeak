# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/MartinPdeS/DeepPeak/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                 |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| DeepPeak/algorithms/amplitude/base.py                |       18 |       18 |        4 |        0 |      0% |      1-40 |
| DeepPeak/algorithms/amplitude/cholesky.py            |       56 |       56 |       14 |        0 |      0% |     1-118 |
| DeepPeak/algorithms/amplitude/closed\_form.py        |       69 |       69 |       16 |        0 |      0% |     1-161 |
| DeepPeak/algorithms/base.py                          |        0 |        0 |        0 |        0 |    100% |           |
| DeepPeak/algorithms/non\_maximum\_suppression.py     |      110 |      110 |       16 |        0 |      0% |     1-315 |
| DeepPeak/dataset.py                                  |       28 |       21 |       16 |        0 |     20% |     28-78 |
| DeepPeak/helper.py                                   |       18 |       11 |        4 |        0 |     32% |     33-48 |
| DeepPeak/machine\_learning/classifier/autoencoder.py |       43 |        1 |        4 |        2 |     94% |84->exit, 93 |
| DeepPeak/machine\_learning/classifier/base.py        |       80 |       28 |       18 |        5 |     60% |27-45, 49-50, 54-55, 59, 63-69, 107, 124-126, 135-136, 205, 211->215, 216-217 |
| DeepPeak/machine\_learning/classifier/dense.py       |       32 |        1 |        6 |        2 |     92% |62->exit, 71 |
| DeepPeak/machine\_learning/classifier/metrics.py     |       48 |       12 |        6 |        0 |     67% |18-34, 104 |
| DeepPeak/machine\_learning/classifier/utils.py       |       38 |       38 |        2 |        0 |      0% |     1-245 |
| DeepPeak/machine\_learning/classifier/wavenet.py     |       40 |        1 |        6 |        2 |     93% |66->exit, 77 |
| DeepPeak/signals.py                                  |      132 |       35 |       42 |       12 |     64% |106, 116, 117->120, 145-153, 165->169, 170, 176->184, 203, 209, 225-233, 280, 288->295, 333-345 |
| DeepPeak/utils.py                                    |       71 |       71 |       14 |        0 |      0% |     1-185 |
| DeepPeak/visualization.py                            |      202 |      202 |       70 |        0 |      0% |     1-800 |
|                                            **TOTAL** |  **985** |  **674** |  **238** |   **23** | **28%** |           |


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