# D-caRbon: 城市建筑集群与电网友好互动降碳潜力评估

## 项目简介 | Overview

本项目使用纯 Python 流程，评估能源受端城市中建筑个体、建筑集群与电网协同互动的降碳潜力。当前示例以澳门数据为主，覆盖建筑数据准备、微气象修正、灵活性计算与碳排放可视化。

This project provides a pure-Python workflow to assess carbon-reduction potential from coordinated interactions among individual buildings, urban building clusters, and the power grid. The current example focuses on Macao, covering data preparation, microclimate correction, flexibility estimation, and carbon-emission visualization.

## 主要功能 | Key Features

- 建筑级建模：温度序列估计、热参数准备、虚拟储能状态计算。
- 集群级建模：建筑密度分析、空调排热导致的微气象温升估计。
- 电网与碳评估：聚合放电功率、碳排放估算及多种图形输出。
- 数据工具：支持自动下载澳门 OSM 建筑数据与天气数据清洗。

- Building-level modeling: temperature estimation, thermal parameter preparation, and virtual storage state calculation.
- Cluster-level modeling: building density analysis and microclimate rise estimation from AC heat rejection.
- Grid and carbon assessment: aggregated discharging power, carbon-emission estimation, and figure export.
- Data utilities: automated OSM building download and weather data fetching/cleaning for Macao.

## 项目结构 | Project Structure

```text
D-caRbon/
├── main.py                          # 主流程入口 | Main pipeline entry
├── requirements.txt                 # Python 依赖 | Python dependencies
├── config/
│   └── default.yaml                 # 默认路径与常量 | Default paths and constants
├── data/                            # 输入与缓存数据 | Input and cached data
├── outputs/                         # 输出结果目录 | Output directory
├── src/
│   ├── building/                    # 建筑级模型 | Building-level models
│   ├── cluster/                     # 集群与微气象 | Cluster and microclimate
│   ├── grid/                        # 电网与碳核算 | Grid and carbon modules
│   ├── data_io/                     # 数据读取与预处理 | Data I/O and preprocessing
│   └── utils/                       # 绘图与辅助工具 | Plotting and helper tools
├── download_osm_macao_buildings.py  # 下载澳门 OSM 建筑 | Download Macao OSM buildings
└── fetch_macao_weather.py           # 抓取并清洗天气数据 | Fetch and clean weather data
```

## 环境要求 | Requirements

- Python 3.8 或更高版本。
- 建议使用虚拟环境（如 `.venv`）。

- Python 3.8 or newer.
- A virtual environment (for example `.venv`) is recommended.

安装依赖 | Install dependencies:

```bash
pip install -r requirements.txt
```

## 快速开始 | Quick Start

### 1) 运行主流程 | Run the main pipeline

```bash
python main.py
```

`main.py` 按流程执行建筑级计算、微气象修正、灵活性聚合与碳排放图形输出。

`main.py` runs the end-to-end pipeline: building-level computation, microclimate correction, flexibility aggregation, and carbon-related visual outputs.

### 2) 可选：下载澳门建筑数据 | Optional: Download Macao building data

```bash
python download_osm_macao_buildings.py
```

### 3) 可选：抓取并清洗天气数据 | Optional: Fetch and clean weather data

```bash
python fetch_macao_weather.py
```

## 配置说明 | Configuration

默认配置位于 `config/default.yaml`，包括数据路径、输出路径和物理常数。可按本地数据组织方式进行修改。

Default settings are in `config/default.yaml`, including data/output paths and physical constants. Adjust them to match your local data layout.

## 典型输出 | Typical Outputs

- 建筑密度 3D 可视化（HTML）
- 微气象温升热力图（HTML/PNG）
- 建筑灵活性时序图（PNG）
- 聚合放电功率对比图与直方图（PNG）

- 3D building-density visualization (HTML)
- Microclimate-rise heatmap (HTML/PNG)
- Building flexibility time-series figures (PNG)
- Aggregated discharging-power comparison plots and histogram (PNG)

## 致谢 | Acknowledgement

本项目用于城市建筑群与电网友好互动降碳潜力研究与方法验证。

This project supports research and method validation for carbon reduction through grid-friendly interactions of urban building clusters.
