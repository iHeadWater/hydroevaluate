<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-29 17:35:04
 * @LastEditTime: 2023-10-29 19:14:44
 * @LastEditors: Wenyu Ouyang
 * @Description: USE AI TO FORECAST FLOOD
 * @FilePath: \AIFloodForecast\README.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# AIFloodForecast

It's a project for flood forecasting based on public data and artificial intelligence technology (especially deep learning). The project is still in progress, and the current version is only a prototype.

## Introduction

The project is based on the [PyTorch](https://pytorch.org/) framework, and the main code is written in Python. 

It is divided into two parts: data processing and model training. The data processing part is currently mainly based on our hydro-opendata project, which is used to download, process, read and write public data source related to flood forecasting. The model training part is mainly based on the [torchhydro](https://www.pytorchlightning.ai/) framework, which is our self-developed framework focusing on hydrological forecasting.

The idea of the project is to use the public data source from data-rich regions such as United States and Europe to train a foundation model. Then we use the trained model to predict river stage or discharge in data-poor regions such as China (actually ther are much data in China, but most are not accessible to the public). The current version is mainly based on Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) model with precipitation from [Global Precipitation Measurement (GPM)](https://gpm.nasa.gov/) and Global Forecast System (GFS) as input and river stage or discharge as output.

## Installation

The project is based on Python 3.10. The required packages are listed in `env.yml`. You can install them by running the following command:

```bash
conda env create -f env.yml
```

## Usage

The project is still in progress, and the current version is only a prototype. The main code is in the root folder. You can run the code by running the following command:

```bash
python main.py
```