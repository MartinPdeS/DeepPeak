# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/MartinPdeS/DeepPeak/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                 |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| DeepPeak/algorithms/non\_maximum\_suppression.py     |      110 |      110 |       16 |        0 |      0% |     1-352 |
| DeepPeak/dataset.py                                  |       28 |       21 |       16 |        0 |     20% |     21-54 |
| DeepPeak/helper.py                                   |       18 |       11 |        4 |        0 |     32% |     33-48 |
| DeepPeak/machine\_learning/classifier/autoencoder.py |       39 |        1 |        2 |        1 |     95% |        62 |
| DeepPeak/machine\_learning/classifier/base.py        |       80 |       28 |       18 |        5 |     60% |25-43, 47-48, 52-53, 57, 61-67, 98, 115-117, 126-127, 190, 196->200, 201-202 |
| DeepPeak/machine\_learning/classifier/dense.py       |       28 |        1 |        4 |        1 |     94% |        44 |
| DeepPeak/machine\_learning/classifier/metrics.py     |       48 |       12 |        6 |        0 |     67% | 16-32, 84 |
| DeepPeak/machine\_learning/classifier/utils.py       |       38 |       38 |        2 |        0 |      0% |     1-249 |
| DeepPeak/machine\_learning/classifier/wavenet.py     |       36 |        1 |        4 |        1 |     95% |        49 |
| DeepPeak/signals.py                                  |      132 |       35 |       42 |       12 |     64% |104, 114, 115->118, 143-151, 163->167, 168, 174->182, 199, 205, 221-229, 276, 284->291, 329-345 |
| DeepPeak/utils.py                                    |       71 |       71 |       14 |        0 |      0% |     1-167 |
| DeepPeak/visualization.py                            |      202 |      202 |       70 |        0 |      0% |     1-767 |
|                                            **TOTAL** |  **830** |  **531** |  **198** |   **20** | **32%** |           |


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