### 命名规范（figures 文件命名）

- 目标：简洁、统一、可读；尽量保留信息但避免冗余。

#### 基本规则
- 全部小写。
- 词与词之间用下划线 `_` 连接：如 `spin_corr`, `power_law`。
- 参数组之间用下划线 `_` 分隔：如 `jk-4_jperp0.1_u18_lx50`。
- 参数键值的连接：
  - 若值为整数，直接紧随其后：`u18`, `lx50`, `jperp1`。
  - 若值为小数或带符号（负号/小数点），使用连字符 `-`：`jk-4`, `delta-0.3`, `t2-0.6`, `pi-0`。
  - t2 参数为避免歧义，统一使用连字符形式，不论整数或小数：`t2-1`, `t2-0.3`（不要写成 `t21`）。目前数据中 t2 > 0。
- 科学计数法统一转为十进制最简表示（不四舍五入，只去除无效的 0）：
  - `1.000000e-01` → `0.1`，`5.000000e-01` → `0.5`，`1.0` → `1`。
- 统一术语：
  - `1D/2D` → `1d/2d`
  - `Ping/NoPing` → `pin/nopin`
- 扩展名小写：`.png`, `.pdf`, `.eps`, `.svg`, ...

#### 参数顺序（建议）
- 主题 + 可选子主题 + 参数组（按照：`jk` → `jperp` → `u` → `lx` → `ly` → `jh` → `t2/t` → `delta` → `pi`）+ 其它标签（如 `1storder/2ndorder`）。

#### 示例
- `singlet_sc_corr_extrapolation_Jk-4Jperp1.000000e-01U18Lx50_2ndorder.png`
  → `singlet_sc_corr_extrapolation_jk-4_jperp0.1_u18_lx50_2ndorder.png`
- `Kondo2LegSpinCorrt20.3U2_pi_0.pdf`
  → `kondo_2leg_spin_corr_t2-0.3_u2_pi-0.pdf`
- `SpinSingleOrbitaldelta=0.2_1D_3x20J=0Pin.svg`
  → `spin_single_orbital_delta-0.2_1d_3x20_j0_pin.svg`
- `PhaseDiagram1D.eps` → `phase_diagram_1d.eps`

#### 警惕项
- 避免连续分隔符：自动合并多余的 `_` 或 `-`。
- 大小写仅在同一文件系统下重命名时可能产生冲突；脚本通过两步改名避免大小写冲突。
- 若最终目标名与现存文件完全同名，则自动追加数字后缀 `-1`, `-2`, ... 以避免覆盖。

#### 适用范围
- 本规范适用于 `plot/*/figures` 下的图文件。
- 若脚本生成的名字与论文/图注强关联，可在图注处注明原始参数；必要时可在文件名前加更具体的语义前缀（不强制）。

#### 变更日志

- **2026-03-05**: `src_kondo_zigzag_ladder/` 的 `vmps.cpp` 和 `measure.cpp` 输出文件名末尾新增 `_OBC` 或 `_PBC` 后缀（由 `params.json` 中的 `Geometry` 字段决定）。此日期之前产生的所有 zigzag ladder 数据均为 OBC，文件名不含几何后缀。MATLAB 绘图脚本已更新，会优先查找带后缀的文件，找不到时自动回退到无后缀的旧文件名。
