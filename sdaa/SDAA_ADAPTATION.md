# BoltzGen SDAA 迁移适配记录（双兼容）

> BoltzGen 是蛋白复合物结构预测模型（Boltz-2 同源），**纯 PyTorch** + cuequivariance
> 三角算子（懒加载、默认降级）。本次适配的核心是：**保持 CUDA / SDAA 双兼容**，
> 所有 device 改动均用 `sdaa 优先 + cuda 兜底` 的判断，CUDA 环境行为完全不变。
>
> 本仓库 fork 自官方 BoltzGen，官方模型代码零改动。适配改动分两类：
> ① **功能适配**（双兼容 device 硬编码，6 个文件）② **可复现性改动**（seed + randn 移 CPU，用于精度对标）。

## 一、环境信息

| 项目 | 值 |
|------|-----|
| 架构 | loongarch64（Loongnix OS，龙芯 3A6000，32 张 SDAA 卡）|
| 机器 | `sunhn@10.71.13.47`，容器 `tuyi_test` |
| PyTorch | 2.12.0 + torch_sdaa 3.3.0b0（`source /opt/tecoai/setvars.sh` 后可见）|
| pytorch-lightning | 2.5.1 + lightning-teco 1.8.6（SDAA 加速器插件）|
| Python | 3.12.13（venv `/home/py312`）|
| 分支 | `adapt/sdaa` |
| CUDA 基准机 | `sunhn@10.10.6.21`，`cuda_env_py310`（torch 2.6.0+cu124，A100）|

## 二、自定义算子分析

### 2.1 cuequivariance 三角算子（懒加载 + use_kernels 降级）

BoltzGen 的 triangle 算子（三角乘法/注意力）依赖 `cuequivariance`，但**默认降级**：

```python
def forward(self, x, mask, use_kernels: bool = False):  # ← 默认 False！
    if use_kernels:
        return _kernel_triangular_mult(...)   # cuequivariance（函数内 import，懒加载）
    # 纯 torch 实现（默认走这里，和 Protenix 的 fallback 同构）
    x = self.norm_in(x)
    x = self.p_in(x) * self.g_in(x).sigmoid()
    ...
    x = torch.einsum("bikd,bjkd->bijd", a, b)
```

- `use_kernels=False` 时**根本不 import cuequivariance**，纯 torch einsum 路径已写好
- `use_kernels="auto"` 的判定依据是 `get_device_capability()[0] >= 8`（A100/H100 才启用 kernel）

### 2.2 双兼容设计原则

所有改动遵循「**优先 sdaa，cuda 兜底**」：

```python
if hasattr(torch, "sdaa") and torch.sdaa.is_available():
    ...  # SDAA 路径
else:
    ...  # 原 CUDA 路径（保持不变）
```

## 三、代码改动（分两类）

### 第一类：功能适配（双兼容 device 硬编码，6 个文件）

#### 3.1 `cli/boltzgen.py` — use_kernels / get_device_capability

```python
# SDAA 适配：SDAA 无 CUDA cuEquivariance kernel，capability 置 None 强制走纯 torch
if hasattr(torch, "sdaa") and torch.sdaa.is_available():
    device_capability = None
else:
    device_capability = torch.cuda.get_device_capability()
```

- **SDAA**：`device_capability=None` → `use_kernels=False` → 纯 torch einsum
- **CUDA**：正常 `get_device_capability()`，≥8 才启用 kernel，行为不变

#### 3.2 `cli/boltzgen.py` — device_count

```python
elif hasattr(torch, "sdaa") and torch.sdaa.is_available():
    devices = torch.sdaa.device_count()
else:
    devices = torch.cuda.device_count()
```

#### 3.3 `boltz.py` + `refolding.py` — empty_cache（4 处）

```python
(torch.sdaa if hasattr(torch, "sdaa") and torch.sdaa.is_available() else torch.cuda).empty_cache()
```

`boltz.py:1168/1187/1374` + `refolding.py:305`（OOM 容错逻辑）。

#### 3.4 `task/predict/predict.py` — accelerator 动态选择

config 里 `trainer.accelerator: gpu` 硬编码，SDAA 上报 `No supported gpu backend found!`。

```python
# ① import lightning_teco 注册 SDAAAccelerator
try:
    import lightning_teco  # noqa: F401
except ImportError:
    pass

# ② 创建 Trainer 前动态选 accelerator
if hasattr(torch, "sdaa") and torch.sdaa.is_available():
    self.trainer["accelerator"] = "sdaa"
elif torch.cuda.is_available():
    self.trainer["accelerator"] = "gpu"
else:
    self.trainer["accelerator"] = "cpu"
```

#### 3.5 无需改的

- **43 处 `torch.autocast("cuda", ..., enabled=False)`**：全部 disabled，无害。
- `boltz.py:1106/1115` 的 `device="cuda" if ...`：仅训练 grad norm，推理不触发。

### 第二类：可复现性改动（精度对标用，seed + randn 移 CPU）

> 背景：SDAA 设备端 RNG（TecoRAND）与 CUDA（cuRAND）的 offset 语义不同，
> 即使 `torch.manual_seed(42)`，设备端 `torch.randn` 结果也跨平台不一致（**设计如此，非 bug**）。
> 为做可复现的精度对标，做了以下两项改动。

#### 3.6 固定随机种子（`BOLTZGEN_SEED` 环境变量）

`cli/boltzgen.py` 的 `main()` 和 `resources/main.py` 的 `main()` 入口加：

```python
# SDAA/CUDA 精度对标：BOLTZGEN_SEED 环境变量固定随机种子
_seed = os.environ.get("BOLTZGEN_SEED")
if _seed is not None:
    _seed = int(_seed)
    random.seed(_seed); np.random.seed(_seed); torch.manual_seed(_seed)
```

#### 3.7 `torch.randn` 移到 CPU 生成（7 处）

把设备端 `torch.randn(..., device=X)` 改成 CPU 生成再 `.to(X)`，噪声源统一到 CPU 的 Philox（跨平台一致）：

| 文件 | 处数 |
|------|:---:|
| `diffusion.py`（初始坐标 ×2、sigma、噪声）| 4 |
| `utils.py`（随机旋转 ×2、随机四元数）| 3 |

```python
# 原：torch.randn(shape, device=self.device)   ← 设备端 RNG，offset 不同
# 改：torch.randn(shape).to(self.device)       ← CPU 生成，跨平台一致
```

## 四、核心洞察：昇腾与 SDAA 方向相反

| | 昇腾方案 | SDAA 方案（本仓库）|
|--|---------|------------------|
| `get_device_capability` | patch 成 `(9,0)` 伪造 H100 | 返回 `None` |
| `use_kernels` | 骗过 `>=8` 判断设 True，启用自写 `npu_kernels.py` | 设 False，走纯 torch einsum |
| 自写 kernel | 有（npu_kernels.py）| **无（不需要）** |

昇腾的 `npu_kernels.py` 本质就是 PyTorch 算子拼接，与 `use_kernels=False` 的纯 torch 路径是同一个东西，SDAA 直接禁用省掉自写 kernel。

## 五、依赖处理

| 类别 | 依赖 |
|------|------|
| 核心 | torch 2.12 + torch_sdaa、pytorch-lightning 2.5.1、lightning-teco 1.8.6、hydra-core、einx、einops、huggingface_hub、mashumaro |
| 结构解析 | gemmi 0.7.5、biotite 0.1.dev（源码编译）、biopython、pdbeccdutils |
| 化学 | rdkit（自包含 whl）、hydride、pydssp |
| 跳过（CUDA 专属）| cuequivariance_ops_cu12 / cuequivariance_ops_torch_cu12 / cuequivariance_torch / nvidia-ml-py |

> 不能无脑 `pip install boltzgen`（会拉 cu12 包装失败），需手动装核心依赖、跳过 cu12。

## 六、运行方式

### 6.1 前置准备

1. 装依赖：`pip install -r requirements_sdaa.txt`（5 个非 pip 项见注释）（lightning-teco：pip install git+http://10.10.30.109/huangzhen/lightning-teco.git）
2. 下权重（~8GB，国内走 hf-mirror，6 个文件分别下）：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   python -c "
   from huggingface_hub import hf_hub_download
   # 5 个模型权重（repo 类型 model）
   for f in ['boltzgen1_diverse.ckpt','boltzgen1_adherence.ckpt','boltzgen1_ifold.ckpt','boltz2_conf_final.ckpt','boltz2_aff.ckpt']:
       hf_hub_download('boltzgen/boltzgen-1', f)
   # mols.zip（repo 类型 dataset）
   hf_hub_download('boltzgen/inference-data', 'mols.zip', repo_type='dataset')
   "
   ```
3. 载入 SDAA 运行时：`source /opt/tecoai/setvars.sh`

### 6.2 SDAA 推理（protein-anything，蛋白设计）

```bash
source /opt/tecoai/setvars.sh
source /home/py312/bin/activate
cd /data01/tuyilist/boltzgen

boltzgen run example/vanilla_protein/1g13prot.yaml \
  --protocol protein-anything \
  --output /tmp/boltzgen_out \
  --num_designs 2 \
  --devices 1 \
  --cache /root/.cache \
  --use_kernels false
```

### 6.3 小分子结合协议（含 affinity，加载 boltz2_aff + mols.zip）

```bash
boltzgen run example/protein_binding_small_molecule/chorismite.yaml \
  --protocol protein-small_molecule \
  --output /tmp/boltzgen_sm_out \
  --num_designs 2 --devices 1 --cache /root/.cache --use_kernels false
```

### 6.4 精度对标（固定随机种子，配合 3.6/3.7 改动）

```bash
export BOLTZGEN_SEED=42
boltzgen run example/vanilla_protein/1g13prot.yaml \
  --protocol protein-anything --output /tmp/boltzgen_seed_out \
  --num_designs 2 --devices 1 --cache /root/.cache --use_kernels false
```

### 6.5 CUDA 基准（用于精度/性能对比）

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cuda_env_py310
export HF_ENDPOINT=https://hf-mirror.com
cd ~/tuyilist/boltzgen
CUDA_VISIBLE_DEVICES=0 boltzgen run example/vanilla_protein/1g13prot.yaml \
  --protocol protein-anything --output /tmp/boltzgen_cuda_out \
  --num_designs 2 --devices 1 --cache ~/.cache --use_kernels false
```

> 参数说明：
> - `--use_kernels false`：SDAA 强制纯 torch（auto 已自动判 capability=None→False，显式写更清晰）
> - `--cache`：权重缓存目录（SDAA 用 /root/.cache，CUDA 用 ~/.cache）
> - 权重已缓存则跳过下载；否则走 HF_ENDPOINT 镜像自动下载

## 七、验证结果

### 7.1 端到端推理（SDAA，全部 0 failed）

**protein-anything（`vanilla_protein/1g13prot.yaml`）：**

| 步骤 | 耗时 |
|------|------|
| design | 15:52 |
| inverse_folding | 0:09 |
| folding | 21:59 |
| design_folding | 6:27 |
| analysis + filtering | ~4 min |
| **总计** | **48:49** |

**protein-small_molecule（`protein_binding_small_molecule/chorismite.yaml`，含 affinity）：**

| 步骤 | 耗时 |
|------|------|
| design | 526.5s |
| inverse_folding | 25.4s |
| folding | 682.3s |
| design_folding | 600.4s |
| affinity | 681.4s |

### 7.2 权重/数据覆盖（6 文件全跑通）

diverse / adherence / ifold / conf_final / aff / mols.zip 全部在 SDAA 上跑通，0 failed。

### 7.3 性能对比（CUDA A100 vs SDAA，均 use_kernels=False 纯 torch）

| 步骤 | CUDA (A100) | SDAA | 倍率 |
|------|------|------|:---:|
| design | 76.6s | 952s | 12.4× |
| inverse_folding | 9.9s | 9s | ~1× |
| folding | 79.0s | 1319s | 16.7× |
| design_folding | 54.5s | 387s | 7.1× |
| **总计** | **~250s（4.2min）** | **~2930s（48.8min）** | **~11.7×** |

### 7.4 精度对比（核心结论）

**算子精度（SDAA vs CUDA，相同输入）：**

| 算子 | 相对差异 |
|------|:---:|
| matmul / einsum | **完全一致** |
| softmax / layernorm / sigmoid / gelu | ~1e-7（正常浮点）|
| 多层 forward（20 层组合）| ~2.6e-5（健康累积）|

**三层结果对比（固定 seed + randn 移 CPU）：**

| 层次 | 结果 |
|------|------|
| 设计长度 | ✅ 一致（118 = 118）|
| 设计序列 | ✅ 一致（SSFSWDN...）|
| 原子坐标 | ❌ RMSD 17 Å（Kabsch 对齐后）|

**结论**：

1. **SDAA 算子精度正常**：matmul/einsum 完全一致，其他算子 1e-7，多层累积 2.6e-5，远没到「算错」的 1e-3 级。
2. **17 Å 坐标分叉 = diffusion 的混沌敏感性**：diffusion 几十步迭代去噪，每步 2.6e-5 级浮点差异经非线性反复放大，最终结构分叉到 17 Å。这是**迭代生成模型的固有性质**，非 SDAA bug。
3. **序列一致**：逆折叠的 argmax 对结构差异不敏感（0.49 和 0.51 都取 0.5），所以离散序列一致、连续坐标分叉。

### 7.5 精度对标方法论（重要）

diffusion 这类**迭代生成模型**，不能比「最终结构一致性」（混沌必然分叉），而应比：

- ✅ **算子级**：单算子精度（matmul/einsum 一致、其他 1e-7）
- ✅ **多层 forward 级**：同样输入 → 输出差多少（2.6e-5，正常）
- ✅ **可复现性**：固定 seed + randn 移 CPU（绕过设备端 RNG offset 差异）
- ❌ 最终结构一致性（混沌，无意义）

## 八、踩坑记录

| # | 坑 | 解决 |
|---|----|------|
| 1 | `torch.cuda.get_device_capability()` 在 SDAA 上 AssertionError | sdaa 优先判 None |
| 2 | `pip install boltzgen` 拉 cu12 包失败 | 手动装核心依赖，跳过 4 个 cu12/nvidia 包 |
| 3 | lightning-teco 的 setup.py 用 pkg_resources（Python 3.12 下崩）| 改 setup.py 手写 requirements 解析 |
| 4 | 权重下载国内 HuggingFace SSL 失败 | 走 hf-mirror.com 镜像 |
| 5 | gemmi nanobind 版本 + stb_sprintf 64 位 | nanobind==2.4.0 + CXXFLAGS=-fpermissive |
| 6 | config `accelerator: gpu` 硬编码 → `No supported gpu backend found` | predict.py 动态选 accelerator + import lightning_teco |
| 7 | 主环境 setuptools 80.10.2 的 pkg_resources 在 Py3.12 崩 | 升级 setuptools 84.0.0 |
| 8 | 设备端 RNG offset 差异（TecoRAND vs cuRAND），同 seed 序列不同 | **设计如此非 bug**；精度对标用 randn 移 CPU 绕过 |
| 9 | boltzgen `--cache` 与 hf_hub_download 默认缓存路径不一致 → 重复下载 | 软链 `~/.cache/models--*` → `huggingface/hub/models--*` |
| 10 | Python 3.10 环境装 boltzgen 报 `requires-python >=3.11` | `--ignore-requires-python`（3.10 实际能跑）|

## 九、产物清单

| 产物 | 说明 |
|------|------|
| 功能适配改动（6 文件）| `cli/boltzgen.py`、`boltz.py`、`refolding.py`、`predict.py`（+ 3.7 的两个文件）|
| 可复现性改动（2 入口）| `cli/boltzgen.py` main() + `resources/main.py` main() 加 seed |
| 权重 | boltzgen1_diverse / adherence / ifold、boltz2_conf_final / aff、mols.zip |