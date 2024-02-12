<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-29 17:35:04
 * @LastEditTime: 2024-02-12 15:49:47
 * @LastEditors: Wenyu Ouyang
 * @Description: Hydro forecast
 * @FilePath: \HydroForecast\README.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# HydroForecast

It's a project for hydrological forecasting based on big data and artificial intelligence technology (especially deep learning). The project is still in progress, and the current version is only a prototype.

## Introduction

The project is based on the [PyTorch](https://pytorch.org/) framework, and the main code is written in Python. 

It is divided into two parts: data processing and model training. The data processing part is currently mainly based on our [hydrodata](https://github.com/iHeadWater/hydrodata) project, which is used to download, process, read and write public data source related to flood forecasting. The model training part is mainly based on the [torchhydro](https://github.com/iHeadWater/torchhydro) and [hydromodel](https://github.com/iHeadWater/hydromodel) framework, which is our self-developed framework focusing on hydrological forecasting

The idea of the project is to use the public data source from data-rich regions such as United States and Europe to train a foundation model. Then we use the trained model to predict river stage or discharge in data-poor regions such as China (actually ther are much data in China, but most are not accessible to the public). The current version is mainly based on Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) model with precipitation from [Global Precipitation Measurement (GPM)](https://gpm.nasa.gov/) and Global Forecast System (GFS) as input and river stage or discharge as output.

## Installation

The project is based on Python 3.10. The required packages are listed in `env.yml`. You can install them by running the following command:

```bash
# simply install a new environment AIFF
conda env create -f env.yml
# then we install packages developed by ourselves as follows
conda activate HydroForecast
# xxx means your Github username; xxxxx means the name of the package; xx means the git-branch of the package
pip install git+ssh://git@github.com/xxx/xxxxx.git@xx
```

The packages we developed are listed as follows in [iHeadWater](https://github.com/iHeadWater):

```bash
torchhydro
hydromodel
```

We'd better use the latest version of the packages. You can check the version of the packages in Github.

## Usage

The project is still in progress, and the current version is only a prototype. The main code is in the root folder. You can run the code by running the following command:

```bash
python main.py
```