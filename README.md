<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-29 17:35:04
 * @LastEditTime: 2024-05-30 09:06:30
 * @LastEditors: Wenyu Ouyang
 * @Description: Hydro forecast
 * @FilePath: \hydroevaluate\README.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# HydroForecast

This project aims to provide a unified evaluation framework for hydrological models, facilitating the evaluation and comparison of different models.

Currently, both physically-based and machine learning-based hydrological models heavily rely on existing datasets for evaluation, without considering the performance of hydrological forecasts. For example, many papers divide the CAMELS dataset into training and testing sets, train models on the training set, and evaluate them on the testing set. However, in actual forecasting, it is common to distinguish between observed rainfall and forecasted rainfall. Models should not have access to any observed data within the forecast period. Therefore, a more realistic evaluation approach would be to evaluate models without using any observed data as input within the forecast period. While the differences may not be significant when comparing different models, this evaluation approach is more appropriate for assessing actual forecasting performance.

Furthermore, there is a lot of research on model evaluation, and we will continuously incorporate relevant studies into the program to explore the topic more comprehensively.