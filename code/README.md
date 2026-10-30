# 运行入口

- `run_all.py`：冒烟复现或完整复现的总入口。
- `generate_data.py`：生成论文图所需的原始数值数据。
- `plot_results.py`：只读取 `data/` 中的新数据并绘图。
- `verify_theory.py`：快速检查理论公式以及两个模拟器是否等价。
- `check_limits.py`：检查 GitHub 的 25 MB 单文件和 100 文件单目录限制。

原始作者脚本保存在 `original_reference/`。该目录不参与新入口的导入和运行，以免历史路径与旧版本互相干扰。

