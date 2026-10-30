# grp-SIS 论文源码（GitHub 整理版）

本仓库对应论文：

> **SIS Epidemic Modelling on Homogeneous Networked System: General Recovering Process and Mean-Field Perspective**, *Applied Mathematical Modelling* 146 (2025) 116188。

仓库只保留源代码、运行说明、算法核对脚本和论文图的参考副本。原始工程中的 `.npy`、`.npz` 模拟结果（约 2.76 GB）以及 `结果.zip`、`结果.7z` 两个重复备份均已移除。读者按下文命令重新运行即可生成数据与图。

## 目录

```text
.
├── README.md
├── 复现说明.md
├── requirements.txt
├── code/
│   ├── sis_model.py            事件驱动 grp-SIS 模拟器
│   ├── generate_data.py        按实验生成原始数值数据
│   ├── plot_results.py         根据新生成的数据绘制 Fig.3–Fig.7
│   ├── verify_theory.py        理论值检查与优化模拟器逐事件等价检查
│   ├── run_all.py              一键冒烟/完整复现入口
│   └── original_reference/     作者原始脚本，未改代码逻辑
├── figures/
│   ├── reference/              原始工程中的论文参考图
│   └── smoke_example/          小规模冒烟复现所生成的示例图
├── provenance/                 原始脚本压缩快照
└── data/                       运行时生成；数据和中间文件不纳入 Git
```

## 环境

已用 Python 3.9.13、Windows 10 验证。建议使用独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 先做冒烟复现

下面的命令使用缩减后的节点数和重复次数，约几十秒即可跑完全流程，并生成 Fig.3–Fig.7。它用于确认安装、路径和数据流正确，不用于与论文中的大样本结果做数值逐项比较。

```powershell
python code\run_all.py --profile smoke
```

输出位于 `data/` 和 `figures/generated/`。`figures/smoke_example/` 是本次整理时尝试该流程留下的示例，不属于论文正式结果。

## 完整复现

完整参数为 `N=2500`、50 次独立重复、`k=4,6,10`；时间演化实验通常取 `T=50`。建议按实验分批运行：

```powershell
# Fig.3、Fig.4、Fig.6 以及 KL 指标
python code\generate_data.py --experiment temporal --profile paper --seed 20250326
python code\plot_results.py --figure 3
python code\plot_results.py --figure 4
python code\plot_results.py --figure 6

# Fig.5(a-c) 与 Fig.5(d)
python code\generate_data.py --experiment prevalence --profile paper --seed 20250326
python code\plot_results.py --figure 5

# Fig.7(a)
python code\generate_data.py --experiment et-lognormal --profile paper --seed 20250326

# Fig.7(b)
python code\generate_data.py --experiment et-powerlaw --profile paper --seed 20250326
python code\plot_results.py --figure 7
```

也可以一次性排队运行，但完整计算量很大：

```powershell
python code\run_all.py --profile paper
```

完整 Monte Carlo 模拟通常需要数小时到数十小时，取决于 CPU、β 和恢复分布参数。可以用 `--nodes`、`--runs` 先缩小规模检查流程；论文级结果请使用 `--profile paper` 的默认参数。

## 复现边界

1. 作者原始代码没有固定随机种子，因此无法保证与当时服务器上保存的某一次随机实验逐点相同，也无法保证重复运行得到完全相同的 PNG 像素。
2. 本整理版固定了默认种子 `20250326`，保证同一机器、同一依赖版本下可重复；不同种子应得到统计一致的曲线和 KL 数量级。
3. 确定性均值场结果可以直接核对。例如指数分布在 `mu=0.5, k=10` 时的阈值是 `0.05`；对数正态 `(mu=0,sigma=1), k=10` 约为 `0.06065`；幂律 `(alpha=4,t0=1), k=10` 为 `0.15`。
4. 整理过程中保留的旧模拟结果曾用于核对 Table 2：9 个 KL 值均与论文保留的小数位一致。按本次要求，这些结果数据没有随仓库提交，读者需用上述代码重新生成。
5. `code/original_reference/` 是作者脚本的逐字节副本，用于溯源。其中部分旧脚本含作者电脑绝对路径、索引顺序依赖或未完成驱动，不应作为复现入口；正式复现请运行 `code/generate_data.py` 和 `code/plot_results.py`。

更详细的方法、参数和验证记录见 [复现说明.md](复现说明.md)。

