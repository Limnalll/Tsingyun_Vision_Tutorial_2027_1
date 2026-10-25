# Project 1: 实现一个简单的EKF Localization

本项目的需要你在简单的场地中实现一个简单的EKF Localization，并对比不同的运动模型和传感器模型的性能。

**截止日期：11.7， 23:59**

## 可以修改的文件

你只需要修改下面四个文件，其他的文件都是只读的：

- `src/motion_models.py`
- `src/beam_model.py`
- `src/likelihood_field.py`
- `src/ekf.py`


## Setup

整个项目是用 [uv](https://docs.astral.sh/uv/) 构建的，下载方式：

- macOS/Linux: `curl -LsSf
https://astral.sh/uv/install.sh | sh`
- Windows: `powershell -c "irm
https://astral.sh/uv/install.ps1 | iex"`

下载后初始化
```bash
uv sync        
```


## Tasks

### Task 1 — Motion Models

你需要完成 `src/motion_models.py`的以下两个部分：

1. **`velocity_motion_model` + `sample_velocity_motion_model`**: 速度运动模型，参见Probabilistic Robotics第五章的同名算法
   
2. **`odometry_motion_model` + `sample_odometry_motion_model`** ： 里程计运动模型，参见Probabilistic Robotics第五章的同名算法


而后，运行
```bash
uv run task1.py          # sample clouds with and without map conditioning
                          # (figures 5.4 / 5.9 / 5.11) -> results/
uv run pytest -k motion  # your checkpoint for this task
```

`results/`中会有两个运动模型在有地图和无地图情况下的采样点云。

---

### Task 2 — Beam Sensor Model

在`src/beam_model.py`中完成:

1. **`p_beam`**：书中第六章提到的四个概率模型（hit/max/rand/short）
   
2. **`learn_intrinsic_parameters`** — 实现书中提到的EM算法，学习传感器的内在参数（`z_hit`, `z_short`, `z_max`, `z_rand`, `sigma_hit`）和最大测距值`z_max`。

之后，运行

```bash
uv run task2.py          # EM run + convergence figure -> results/;
                          # learned parameters -> data/beam_params_learned.json
uv run pytest -k beam    # checkpoint
```

`results/`中会有EM算法的收敛图。

---

### Task 3 — Likelihood Field Sensor Model
在`src/likelihood_field.py`中实现
**`likelihood_field_model`** 



```bash
uv run task3.py           
uv run pytest -k likelihood
```

`results/`中会有likelihood field model的belief map。

---

### Task 4 — The full stack: EKF localization (chapter 7)

在`src/ekf.py` 中实现EKF Localization 算法

1. **`ekf_predict`** ：EKF的预测部分
2. **`ekf_correct`** ： EKF 的measurement update部分

```bash
uv run task4.py   # 运行pygame程序
```
观察两个传感器模型在正确性和帧率上的不同。

---

## 提交作业

- 请fork这个仓库，并通过PR的方式提交。请用好`.gitignore`，建议完成一个task提交一个commit，不要提交任何中间文件。
- 请在下发的飞书问卷中，提交一个指向清华云盘的链接，包含`results/`的结果和`task4`运行的视频录屏，包括
  - 用beam model的运行轨迹
  - 用likelihood field model的运行轨迹（按2）
  - 两个模型下运行多个不同起始点（按K）
- 这次作业基本上就是对书中已有算法伪代码的实现，没有需要你自己设计的代码或算法，并且在所有的TODO下都有比较完整的伪代码hint，所以本次作业**严禁使用Agent直接生成代码，一旦我们发现你的代码里有AI痕迹，将扣除你的全部成绩**，不过你可以问AI某个函数如何调用等等，只要不直接生成代码，都是允许的。
- Grading：过Github CI 占 60%， 提交的结果占 40%，请务必在截止日期前提交。


