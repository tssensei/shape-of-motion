# 3D Modal Gaussian 项目计划

## 目标

用动态 3D Gaussian Splatting 做 Davis et al. 2015 论文 "Image-Space Modal Bases for Plausible Manipulation of Objects in Video" 的 3D 版本。

长期目标是：从单目视频中恢复、编辑和合成合理的物体运动，把 Gaussian center 的 3D 轨迹看作 Davis 论文里 image-space displacement signal 的 3D 对应物。

当前实验起点是 Shape of Motion, 简称 SOM。SOM 已经提供了动态 3DGS 风格的表示和运动重建能力，但它默认的运动表示还不是我们最终想要的 modal analysis 表示。

## 关键论文及其作用

### Davis 2015

Davis 从小幅 image-space motion 中提取 modal basis：

```text
2D displacement signal over time -> temporal FFT -> frequency peaks -> image-space mode shapes
```

对本项目有用的思想：

- 用观测到的 displacement signal 识别 resonant frequencies。
- 把某个频率上的空间切片看作 mode shape。
- 支持 complex modes，因此不同像素可以有不同相位。
- 用恢复出的 modal basis 做运动编辑和合理的动态模拟。

主要限制：

- Davis 工作在 image space，并假设 weak perspective 和 small motion。它不能直接处理 moving camera，也不能天然分离 camera pose error 和真实物体运动。

### ModalNeRF

ModalNeRF 把 Davis 的思想推进到 3D。它在动态 radiance field 中使用 Lagrangian particles：

```text
3D particle displacement over time -> FFT -> modal analysis -> modal synthesis
```

对本项目有用的思想：

- 运动最好在 persistent 3D points 上分析，而不是在像素上分析。
- rest/canonical space 很重要。
- 小幅周期或准周期运动是合适目标。
- 大 frame-to-frame motion、view-dependent effects、非周期运动都是容易失败的情况。

主要限制：

- ModalNeRF 使用的是 NeRF/particle representation，而我们希望使用 3DGS。

### AmbientGaussian

AmbientGaussian 是第一阶段建模最接近的实现模板。它不是先从已知 displacement 中分析出 modal basis，而是直接给每个 Gaussian 的运动轨迹加上固定的 temporal function basis，尤其是 DCT：

```text
Delta mu_i(t) = sum_k c_i,k * DCT_k(t)
```

这里 basis functions 是固定的，训练真正学习的是每个 Gaussian 的 coefficients。

对本项目有用的思想：

- 不需要在训练前已经有 displacement signal。
- 用 function-constrained trajectories 代替 unconstrained per-frame motion。
- DCT 提供平滑、低频、适合 ambient plant motion 的轨迹表示。
- 加入 rigidity regularization，使相邻 Gaussians 的运动更一致。

主要限制：

- DCT 不是物理 modal basis。它是实用的 trajectory parameterization。

### Shape of Motion

SOM 当前用 low-rank learned SE(3) motion bases 表示 foreground Gaussian motion：

```text
transform_i(t) = sum_k weight_i,k * learned_SE3_basis_k(t)
```

每个 Gaussian 有一组 softmax weights，用来混合若干 learned motion bases。motion bases 本身是随时间变化的 transforms。

这个表示对 reconstruction 有用，但对 Davis-style modal analysis 不理想，因为：

- basis 是为了 rendering 学出来的，不一定有 modal interpretability。
- basis 是 low-rank 的，但不和明确频率或 oscillator behavior 绑定。
- Gaussian center trajectories 可能包含 camera residuals 或 reconstruction artifacts。

## 核心概念

我们想做的 3D Davis 对应关系是：

```text
Davis:
pixel p has 2D displacement u_p(t)

This project:
Gaussian i has 3D center displacement u_i(t)
```

对一个训练好的模型：

```text
u_i(t) = x_i(t) - x_i_rest
```

然后可以对 Gaussian displacement signals 做 modal analysis：

```text
U_i(f) = FFT_t[u_i(t)]
```

某个频率峰值 `f_k` 对应一个 3D mode shape：

```text
phi_k = { U_i(f_k) for all Gaussian i }
```

这就是 Davis-style modal basis extraction 的真正 3D 版本。

## 重要澄清

有两件事情很容易混淆，必须分开。

### Function-Constrained Trajectory Learning

这个阶段不需要预先已有 displacement signal。

例子：

```text
Delta mu_i(t) = sum_k c_i,k * B_k(t)
```

其中 `B_k(t)` 可以是 DCT、sin/cos，或者其他 temporal basis。

模型通过 rendering losses 学习 `c_i,k`。

### Modal Analysis-Derived Spatial Bases

这个阶段需要 displacement signal。

例子：

```text
Delta mu_i(t) = sum_k phi_k,i * q_k(t)
```

其中 `phi_k,i` 是第 `k` 个 spatial mode shape 在 Gaussian `i` 上的值。

这些 mode shapes 必须从观测到的轨迹或已经训练出的轨迹中估计出来。

因此推荐路线是：

```text
First learn stable function-constrained Gaussian trajectories.
Then analyze those trajectories to get modal bases.
Then put the modal bases back into the model for editing/synthesis.
```

## 推荐开发路线

### Stage 0: 稳定 Camera Pose 和 Scale

在做任何 modal analysis 之前，必须先解决相机和尺度问题。

当前推荐路径：

```text
branch megasam1
MegaSAM pose -> scaled DROID-style droid_recon.npy
training camera_type=droid_recon
depth supervision=aligned_depth_anything
```

原因：

- SOM 的训练数据 contract 保持不变。
- 唯一替换的是 camera pose estimator。
- depth supervision 继续使用原 SOM pipeline 的 `aligned_depth_anything`。
- pose translation 已经对齐到 `aligned_depth_anything` 的尺度。

诊断指标：

- 检查 camera path length / median scene depth。
- 检查 training-camera replay。
- 检查 fixed-camera temporal replay，看 foreground motion 是否合理。
- 确认 background 不会作为 camera error 的替代物而移动。

### Stage 1: AmbientGaussian-Style Trajectory Model

替换或绕过 SOM 当前的 low-rank learned SE(3) motion bases，改成直接的 Gaussian trajectory model。

第一版先只建模 Gaussian center translation：

```text
mu_i(t) = mu_i^0 + Delta mu_i(t)
Delta mu_i(t) = sum_k c_i,k * B_k(t)
```

其中：

- `mu_i^0` 是 canonical Gaussian center。
- `B_k(t)` 是固定 temporal basis。
- `c_i,k` 是每个 Gaussian、每个坐标方向要学习的 coefficient。

`B_k(t)` 的可选方案：

1. DCT basis，跟 AmbientGaussian 一致。
2. Sin/cos harmonic basis，频率手动选择几个低频。
3. Sin/cos harmonic basis，频率从 2D tracks 或前一个模型中估计。

第一版应优先使用 DCT，因为简单、稳定、容易验证。

训练 losses：

- RGB rendering loss。
- Mask loss，如果 masks 稳定。
- Depth loss，如果 depth scale 稳定。
- Track reprojection / track consistency loss，如果 tracks 可用。
- Local rigidity 或 neighbor smoothness loss，约束相邻 Gaussian 运动。
- Temporal smoothness loss，约束 displacement 或 velocity。

预期输出：

- 一个动态 3DGS，Gaussian centers 按平滑的 function-constrained trajectories 运动。
- 可以导出的 per-Gaussian trajectories `mu_i(t)`。

### Stage 2: 提取 Gaussian Center Trajectories

Stage 1 训练完成后，导出 foreground Gaussian trajectories：

```text
x_i(t), i = 1..N, t = 1..T
```

分析前需要过滤 Gaussians：

- 只保留 foreground。
- 保留 high opacity / high accumulated visibility 的 Gaussians。
- 保留时间上稳定的 Gaussians。
- 排除明显 background。
- 排除明显受 pruning/splitting artifacts 影响的 Gaussians。

定义 rest position：

```text
x_i_rest = median_t x_i(t)
```

如果 canonical center `mu_i^0` 可靠，也可以直接用它。

计算 displacement：

```text
u_i(t) = x_i(t) - x_i_rest
```

做 modal analysis 前，应去掉 residual global rigid drift：

```text
for each t:
  fit rigid transform from x_i_rest to x_i(t)
  subtract the rigid component
  keep non-rigid residual displacement
```

这一步很重要，因为 camera pose error 否则可能成为最强的 mode。

### Stage 3: Davis-Style 3D Modal Analysis

计算时间频谱：

```text
U_i(f) = FFT_t[u_i(t)]
```

全局频谱能量：

```text
P(f) = sum_i visibility_i * ||U_i(f)||^2
```

从 `P(f)` 的 peaks 中选择 dominant frequencies。

对每个选中的频率 `f_k`，定义 complex 3D mode shape：

```text
phi_k,i = U_i(f_k)
```

complex phase 很有用：

- 不同叶片或枝条可以有相位滞后。
- 可以表示 traveling-wave-like motion。
- 这和 Davis 的 complex mode shape 思想一致。

保存 modal basis：

```text
frequencies: (K,)
mode_shapes: (K, N, 3), complex
rest_centers: (N, 3)
gaussian_ids: (N,)
```

### Stage 4: Modal-Constrained Gaussian Model

把 learned DCT trajectory 替换成 modal basis motion：

```text
Delta mu_i(t) = Re(sum_k phi_k,i * z_k(t))
```

其中 `z_k(t)` 是 scalar 或 complex modal coordinate。

如果只是 replay：

```text
z_k(t) = a_k * exp(j * 2*pi*f_k*t)
```

如果做物理编辑：

```text
q_k'' + 2*zeta_k*omega_k*q_k' + omega_k^2*q_k = f_k(t)
```

这会给出真正 Davis-style 的 3D editing interface：

- 用户在某个点或区域施加 force。
- force 被投影到 modes 上。
- modal coordinates 随时间演化。
- Gaussian centers 由 mode shapes 驱动产生 displacement。
- 场景通过 3DGS 渲染出来。

### Stage 5: Evaluation And Ablations

有价值的对比：

1. SOM 原始 low-rank motion bases。
2. DCT Gaussian trajectory model。
3. Sin/cos frequency trajectory model。
4. Extracted modal replay。
5. Modal editing / synthesis。

指标和诊断：

- Visual replay quality。
- Background stability。
- Track reprojection error。
- Depth consistency。
- Temporal smoothness。
- Extracted motion 的 frequency concentration。
- Fixed camera 下 foreground motion 是否仍然合理。
- 编辑后的运动是否局部合理，是否错误带动无关枝条。

## 对当前 Repo 的实现提示

可能相关的 SOM 文件：

- `flow3d/params.py`
  - 当前的 `MotionBases` 和 Gaussian parameters。
- `flow3d/scene_model.py`
  - 把 motion transforms 应用到 Gaussian means/quats。
- `flow3d/init_utils.py`
  - 当前从 tracks 初始化 motion bases。
- `flow3d/trainer.py`
  - losses 和 optimization。
- `flow3d/data/casual_dataset.py`
  - custom data loading、camera pose、depth、tracks。

第一版代码原型应尽量小心，不要直接删除 `MotionBases`：

1. 新增一个 trajectory module，而不是重写现有 motion basis。
2. 增加一个 config switch：

```text
trajectory_type = som_basis | dct_center | harmonic_center | modal_center
```

3. 保持 DROID-style camera loading 不变。
4. 使用 branch `megasam1` 作为推荐 camera preprocessing baseline。

## 开放风险

### Gaussian Identity

Gaussian centers 不一定是真正的 material points。densification、pruning 和 optimization 都可能改变某个 Gaussian 的含义。

缓解方式：

- 只分析稳定 Gaussians。
- 等 densification 稳定后再做 modal analysis。
- 在 trajectory extraction 前考虑冻结 Gaussian 数量。

### Camera Error

Pose error 可能主导 displacement。

缓解方式：

- 使用 branch `megasam1`。
- 渲染 fixed-camera videos。
- FFT 前去掉 residual global rigid motion。

### Non-Periodic Wind Motion

风吹灌木可能是 quasi-periodic 或 stochastic，不一定是干净的 harmonic oscillator。

缓解方式：

- 先使用 DCT 或较宽的 harmonic basis。
- 不要过早强制使用很少的 modes。
- 把 modal analysis 看作 interpretable approximation，而不是 ground truth physics。

### Spatial Coupling

如果两个独立枝条有相同频率，global mode 可能错误地把它们耦合起来。

缓解方式：

- 使用 masks 或 spatial clusters。
- 按 branch/region 分别提取 modes。
- 允许相同或相近频率下存在多个 spatial modes。

## 近期步骤

1. 使用 branch `megasam1` 训练 `bush2`。
2. 检查 training-camera replay 和 fixed-camera replay。
3. 如果 camera/foreground motion 合理，实现从 SOM checkpoints 导出 Gaussian trajectories。
4. 对 Gaussian center trajectories 做 post-hoc FFT analysis。
5. 并行设计一个 DCT center trajectory module，作为第一个 AmbientGaussian-style replacement for SOM motion bases。

