# 能源受端城市建筑集群与电网友好互动降碳潜力评估

广东省基础与应用基础研究基金区域联合基金青年基金项目。纯 Python 实现，从建筑个体到城市集群再到电网协同的降碳潜力评估。

## 项目结构

```
C DRcarbon/
├── README.md
├── requirements.txt
├── config/default.yaml    # 路径、常数、城市边界等配置
├── data/                  # 建筑、气象等数据目录
├── src/
│   ├── building/          # 建筑个体：热力学、温控负荷、用户行为、灵活性
│   ├── cluster/           # 城市建筑集群：几何、微气象、集群灵活性
│   ├── grid/              # 电网与降碳：电网模型、可再生能源、协同、碳核算
│   ├── data_io/           # 数据获取与预处理
│   └── utils/             # 绘图等工具
├── main.py                 # 主函数入口（分三章）
└── outputs/                # 结果与图输出
```

## 运行前准备

- Python 3.8+
- 安装依赖：`pip install -r requirements.txt`
- 在 `config/default.yaml` 中配置数据路径与参数

## 运行方式

在项目根目录执行：

```bash
python main.py
```

主函数按三章顺序执行：第一章 建筑个体 → 第二章 建筑集群 → 第三章 电网与降碳。
