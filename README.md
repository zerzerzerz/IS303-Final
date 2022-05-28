# README
## Project Code for IS303 Final Project
## Report
- [LaTeX for SJTU](https://latex.sjtu.edu.cn/read/zhdcfcgndpzv)
## Run
- `python get_car_type.py`
  - 获取二手车的品牌信息
- `python get_car_id.py`
  - 获取二手车的id，需要使用二手车的品牌信息辅助查询
- `python get_car_info.py`
  - 根据id查询二手车详细信息
- `pre_process.ipynb`
  - 数据清洗，格式转换，关键词提取
  - 打开该notebook并执行即可获得清洗完成之后的数据
- `python main.py`
  - 训练并测试神经网络，详细配置可以运行`python main.py -h`查看各种命令行参数

## Data
- `data_new`和`data`下面都存放着数据
  - `data`
    - 是在服务器A上爬取的，该服务器网络不稳定，因此爬取了400个批次之后便手动将其停止了
  - `data_new`
    - 是在服务器B上爬取的，该服务器网络较为稳定，因此爬取到了更多的数据
- 下面将介绍`data_new`下面的数据
  - `data_new/car_id3.json`
    - 存放着二手车的ID信息，共64186条
  - `data_new/car_type2.json`
    - 每个汽车品牌都对应着一个url，用来细化搜索范围
  - `data_new/car_type_refinement.json`
    - 数据清洗之后，将每个汽车品牌对应一个整数，这个整数在神经网络中将通过`torch.nn.Embedding()`转换为一个word embedding
  - `data_new/car_info_requests_*.csv`
    - 是分批下载的二手车详细信息，因为需要下载的数据量很大，总共需要下载25小时，因此分批下载并保存，每次下载100辆汽车的信息，一共有0-641共642个批次
  - `data_new/data.csv`
    - 聚合`data_new/car_info_requests_*.csv`并清洗之后得到的数据，用于训练和测试
  - `data_new/data_train.csv`
    - 将`data_new/data.csv`进行shuffle，取90%作为训练集
  - `data_new/data_test.csv`
    - 打乱后的`data_new/data.csv`取剩余10%作为测试集

## Code
- `dataset`
  - 存放着用于加载和归一化数据的数据结构
- `model`
  - 存放着神经网络模型，包含vanilla MLP和skip-connection MLP