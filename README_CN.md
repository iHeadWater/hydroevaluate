
# AIFloodForecast

## 使用方法


1.创建AIFF虚拟环境
```bash
conda env create -f env.yml
```
2.激活环境
```bash
conda activate AIFF
```
3.安装torchhydro
```bash
pip install git+ssh://git@github.com/iHeadWater/torchhydro.git@dev
```
4.对照v001.yml创建新的yml文件（如v002.yml）
```bash
# record version info here as the following format:
data_cfgs:
  sub: "/v002"
  source: "GPM_GFS"
  source_path: "gpm_gfs_data"
  source_region: "US"
  download: 0
  ctx: [0]
  dataset: "GPM_GFS_Dataset"
  sampler: "WuSampler"
  scaler: "GPM_GFS_Scaler"

model_cfgs:
  model_name: "SPPLSTM"
  model_hyperparam:
    seq_length: 168
    forecast_length: 24
    n_output: 1
    n_hidden_states: 80

training_cfgs:
  train_epoch: 50
  save_epoch: 1
  te: 50
  batch_size: 256
  loss_func: "RMSESum"
  opt: "Adam"
  lr_scheduler: {1: 1e-4, 2: 5e-5, 3: 1e-5}
  which_first_tensor: "sequence"
  
train_period: ["2016-08-01", "2016-12-31"]
test_period: ["2016-08-01", "2016-12-31"]
valid_period: ["2016-08-01", "2016-12-31"]

gage_id:
  - '21401550'

var_out: ["streamflow"]
var_t: ["tp"]
```
5.在main.py中最后一行调用yml文件
```bash
run_normal_dl(cfg_path_dir + "v002.yml")
```
6.进入到torchhydro源代码（data_source_gpm_gfs.py）修改数据路径（有三处地方需要修改）
```bash
#os.path.join中修改为读取流量nc文件的路径
def read_streamflow_xrdataset(...):
  ...
  streamflow = xr.open_dataset(
            os.path.join())
  ...
```
```bash
#os.path.join中修改为读取降水nc文件的路径
def read_gpm_xrdataset(...):
  ...
  for basin in gage_id_lst:
            gpm = xr.open_dataset(
                os.path.join())
...
```
```bash
#os.path.join中修改为读取attributes的nc文件的路径
def read_attr_xrdataset(...):
  ...
  attr = xr.open_dataset(
            os.path.join())
  ...
```
