# 从零开始完成 LeRobot + PiperX 实机项目

这不是一份只列主题的阅读清单，而是一份可以照着执行的入门教程。默认你会 Python、PyTorch 和基本深度学习，但没有学过机器人学、ROS、SLAM、相机、ACT，也没有操作过真实机械臂。不要因为暂时看不懂术语而跳过基础；实机实验中，知道“一个数代表什么、一个命令会让什么运动”比先跑出 loss 更重要。

## 使用方法

每个阶段都按这个顺序进行：

1. 阅读本阶段列出的官网页面，只读指定章节。
2. 用自己的话写下术语和数据流，不要只复制文档。
3. 执行命令或小实验，把终端输出、截图、配置和结论保存到 `notes/`。
4. 对照“通过标准”。没有通过就停在本阶段，先解决问题。
5. 进入下一阶段前，把本阶段产生的文件提交到 Git。

建议周期为 8 周，每周工作日每天 2–3 小时、周末每天 4–6 小时。实验室排期不确定时，先完成离线部分；不要为了赶进度压缩安全测试。

---

## 总目标和最终成果

你最终要完成的是下面这条闭环：

```text
Linux/相机/机器人基础
  → 安装 LeRobot
  → 看懂 LeRobotDataset
  → 在 PushT 仿真中训练 ACT
  → 认识 PiperX 的驱动和安全边界
  → 只读连接、低速控制、遥操作
  → 采集并检查 PiperX 数据
  → 训练自己的 ACT
  → 分阶段部署到实机
  → 10 次可复现评估和失败分析
```

最终应得到：

- `notes/`：每一天的学习记录和问题答案。
- `experiments/`：命令、配置、训练日志和评估表。
- 一份 PushT ACT baseline。
- 一份可加载、可视化、可 replay 的 PiperX 数据集。
- 至少两个 PiperX policy checkpoint。
- 一份至少 10 个 episode 的实机评估表和失败视频/日志索引。
- 一份说明 PiperX 如何接入 LeRobot 的技术笔记。

第一项任务固定为：**把一个大号方块从固定区域抓起，放入旁边固定的容器**。初期不做堆叠、移动目标、双臂协作、长轨迹和无人值守。

---

## 第 0 周：建立最基本的机器人直觉

### 0.1 你需要先理解的词

用自己的话写一页笔记，至少解释这些词：

| 词 | 你要理解的意思 |
| --- | --- |
| 关节（joint） | 机械臂中可以转动或移动的一节；每个关节通常有角度、速度、限位 |
| 自由度（DOF） | 可以独立控制的运动数量；PiperX 的准确值要以手册为准 |
| 末端执行器（end effector） | 机械臂末端的夹爪、吸盘或工具 |
| joint space | 用每个关节的位置描述姿态 |
| Cartesian/EE space | 用末端位置和方向描述姿态 |
| 状态（state） | 机器人当前反馈，例如关节位置、速度、夹爪开度 |
| 动作（action） | 发送给机器人要求它执行的控制量 |
| 控制周期/FPS | 每秒读取或发送多少次；控制频率和相机 FPS 不一定相同 |
| episode | 从一个初始状态开始到任务结束的一段完整轨迹 |
| teleoperation | 人直接控制机器人，常用于采集示范数据 |
| policy | 根据观察预测动作的模型 |
| calibration | 建立传感器、关节零位、工具坐标和控制数值之间的对应关系 |

### 0.2 推荐的基础材料

不要求一次学完整本教材，只看概念：

- [Modern Robotics: Introduction](https://modernrobotics.northwestern.edu/chapters/chapter-1-introduction/)，看 Chapter 1 的 rigid body、configuration、degrees of freedom。
- [Modern Robotics: Chapter 2](https://modernrobotics.northwestern.edu/chapters/chapter-2-spatial-velocities-and-twists/)，只看 frame、rotation、translation 的直觉。
- [ROS 2 Documentation: Concepts](https://docs.ros.org/en/jazzy/Concepts.html)，只了解 node、topic、message、service、action；本计划不要求你先学会 ROS。
- [OpenCV Python tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)，看读取摄像头、图像尺寸、颜色空间和保存图片。

### 0.3 小练习

- [ ] 用 Python/NumPy 画一个二维坐标系，说明 x、y、角度正方向。
- [ ] 用 OpenCV 打开一张图片，打印 `shape`、`dtype`，保存一张缩放图。
- [ ] 观看一段机械臂视频，指出“关节运动”“末端运动”“夹爪动作”。
- [ ] 写 300 字说明：为什么“相机画面变了”不等于“机器人知道自己该怎么动”。

通过标准：你能解释 state 和 action 的区别，并能读懂 `numpy.ndarray` 图像的高、宽、通道顺序。

---

## 第 1 周：安装环境和认识 LeRobot

### Day 1：检查机器

官网：

- [Installation](https://huggingface.co/docs/lerobot/main/installation)
- [Compute Hardware Guide](https://huggingface.co/docs/lerobot/main/hardware_guide)

执行：

```bash
cd /home/scarramcci/Project/lerobot
uv sync --locked --extra test --extra dev
ffmpeg -version
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
git rev-parse HEAD
nvidia-smi
```

记录：Python、PyTorch、CUDA、NVIDIA Driver、GPU 显存、LeRobot commit。不要只写“能用”，把完整版本保存到 `notes/day01-environment.md`。

通过标准：`torch.cuda.is_available()` 为 `True`，能导入 `lerobot`，知道 FFmpeg 用来做什么。

### Day 2：第一次看 CLI

官网：[Cheat sheet](https://huggingface.co/docs/lerobot/main/cheat-sheet)。

```bash
uv run lerobot-info --help
uv run lerobot-train --help
uv run lerobot-record --help
uv run lerobot-teleoperate --help
uv run lerobot-dataset-viz --help
```

你现在不需要记参数。请画出：

```text
命令行参数 → config dataclass → dataset/robot/policy → 输出日志或动作
```

阅读仓库：

- `src/lerobot/scripts/`：命令入口。
- `src/lerobot/configs/`：参数和配置。
- `src/lerobot/datasets/`：数据集。
- `src/lerobot/policies/`：模型。
- `src/lerobot/robots/`、`cameras/`、`teleoperators/`：实机边界。

通过标准：能指出 `lerobot-train`、`lerobot-record`、`lerobot-teleoperate` 分别负责什么。

### Day 3：认识数据集

官网：[LeRobotDataset v3](https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3)，只读 format design、load dataset、episodes、timestamps、features。

练习：

- [ ] 找到一个公开 LeRobot 数据集，例如 `lerobot/pusht`。
- [ ] 打印 metadata、episode 数、FPS、feature 名称和 shape。
- [ ] 随机读取一个 sample，分别打印图像、state、action、task、timestamp。
- [ ] 解释为什么不能跨 episode 取时间窗口。

重点理解：训练样本不是一张独立图片，而是“某个 episode 中某个时间点的观察，以及之后要执行的动作”。

通过标准：你能回答“模型输入是什么、监督标签是什么、它们的时间关系是什么”。

### Day 4：认识相机

官网：[Cameras](https://huggingface.co/docs/lerobot/main/cameras)。

学习：USB 摄像头、RGB 图像、分辨率、FPS、曝光、白平衡、视野、遮挡、压缩和时间戳。

练习：

```bash
uv run lerobot-find-cameras
```

如果当前没有相机，使用电脑摄像头或手机拍摄的视频完成练习。记录：

- 相机设备索引和实际物理设备的对应关系。
- `640x480` 与 `1280x720` 对显存和细节的影响。
- 相机位置改变后，训练好的 policy 为什么可能失效。

通过标准：你能从一帧画面判断相机是否看见目标、夹具和接触过程，并知道 FPS 不代表推理 FPS。

### Day 5：认识训练入口

官网：[Imitation Learning for Robots](https://huggingface.co/docs/lerobot/main/il_robots)，先看 introduction、train a policy 和 evaluation 的概念。

阅读 `src/lerobot/scripts/lerobot_train.py`，只追踪：配置如何进入训练、dataset 如何加载、policy 如何创建、checkpoint 如何保存。不要试图读完所有模型代码。

输出：`notes/day05-training-flow.md`，包含一张不超过 12 个节点的流程图。

通过标准：能区分 dataset、policy、optimizer、checkpoint 和 evaluation。

### 周末：安装验收

- [ ] 运行项目中一组不依赖实机的快速测试：`uv run pytest tests -q --maxfail=1`；若依赖缺失，记录具体原因。
- [ ] 建立 Git 分支或至少提交 `notes/day01` 到 `day05`。
- [ ] 写一页“我还不理解的术语”，不要假装已经理解。

---

## 第 2 周：从零理解 ACT 和训练流程

### Day 6：先补序列建模直觉

复习：监督学习中的输入、标签、batch、epoch、train/eval、overfitting、normalization。

把机器人任务写成序列：

```text
观察 o_t = 图像 + 机器人状态
动作 a_t = 关节/末端的控制量
数据 = (o_0, a_0), (o_1, a_1), ...
```

思考并记录：

- 只用当前图像预测一步动作有什么问题？
- 为什么动作需要连续而不能每帧随机跳动？
- 为什么同一张图像可能对应不同动作？例如接近目标和离开目标。

### Day 7：学习 ACT

官网：[ACT](https://huggingface.co/docs/lerobot/main/act)，重点看 architecture、action chunking、training 和 inference。

只需掌握这些直觉：

- image encoder 把图像变成特征；
- state 告诉模型当前关节/末端状态；
- Transformer 根据观察预测一段未来动作，而不是只预测一个动作；
- action chunk 可以减少逐帧抖动，但要求训练和部署的时间窗口一致；
- `forward()` 用于训练并计算 loss，`select_action()` 用于部署时产生动作。

画出：

```text
image_t + state_t → encoder/transformer → [a_t, a_t+1, ... a_t+K]
```

通过标准：不用公式也能说明 action chunk、训练 loss 和实机成功率为什么不是同一个指标。

### Day 8：训练第一个 smoke test

官网：[Train a policy](https://huggingface.co/docs/lerobot/main/il_robots#train-a-policy)。

先使用公开 PushT 数据，设置很小的 steps，目标只是检查数据、模型、GPU 和 checkpoint 能否工作。执行前把完整命令保存到 `experiments/pusht-smoke/command.txt`。

检查：

- 程序是否找到数据集；
- batch 中的图像和 state shape 是否正确；
- loss 是否是有限数；
- GPU 显存是否正常；
- checkpoint 目录包含什么文件。

### Day 9：理解 processor

官网：[Introduction to Processors](https://huggingface.co/docs/lerobot/main/introduction_processors)。

必须理解：

- observation processor：机器人观测进入模型前的转换；
- policy processor：模型输出变成机器人动作前的转换；
- normalize、resize、batch、device、rename 的作用；
- 训练和部署必须使用相同的约定。

练习：追踪一个图像 key 从 dataset 到 policy 的名称、尺寸和 dtype 变化。

### Day 10：动作表示

官网：[Action Representations](https://huggingface.co/docs/lerobot/main/action_representations)。

写表格比较：

| 表示 | 例子 | 优点 | 风险 |
| --- | --- | --- | --- |
| 关节绝对位置 | 每个关节目标角度 | 直观 | 单位、零位和限位必须一致 |
| 关节增量 | 当前角度加一个变化量 | 小动作较安全 | 累积误差和频率敏感 |
| 末端位置/姿态 | x/y/z/旋转 | 任务直观 | 需要运动学和控制器支持 |

暂时不要猜 PiperX 属于哪一种；标成“待 SDK 确认”。

### 周末：PushT 正式 baseline

- [ ] 完成一次 5,000 steps 左右的 ACT 训练，具体步数根据机器时间调整。
- [ ] 保存至少两个 checkpoint、训练配置、loss 曲线和训练耗时。
- [ ] 进行至少 20 个 episode 的评估。
- [ ] 改变一个变量（例如 steps 或 batch size）再做一组对照。
- [ ] 记录成功率、失败原因和显存峰值。

通过标准：你能从命令行一路解释到 checkpoint，并能说明结果差异是由哪个变量造成的。

---

## 第 3 周：认识 PiperX，先学会安全地“不让它乱动”

### Day 11：建立硬件档案

仓库目前没有可确认的 `piperx` 原生配置，因此不能套用 SO-101 或其他机械臂命令。现场填写：

| 项目 | 结果 | 证据位置 |
| --- | --- | --- |
| 精确型号和固件 | 待填写 | 铭牌/截图 |
| 自由度、关节顺序、单位 | 待填写 | 手册/SDK |
| 控制柜、电源和额定参数 | 待填写 | 铭牌/负责人 |
| 串口/CAN/以太网设备 | 待填写 | 系统信息 |
| SDK/ROS2 包和版本 | 待填写 | 安装文件/命令 |
| 位置、速度、力矩模式 | 待填写 | SDK 文档 |
| 状态反馈字段和频率 | 待填写 | API 采样 |
| 软限位、硬限位、碰撞检测 | 待填写 | 手册/验证 |
| 末端夹具、遥操作设备 | 待填写 | 现场盘点 |
| 急停、断能、故障恢复 | 待填写 | 负责人演示 |

产出：`notes/piperx-inventory.md`。任何不知道的项目写“未知”，并写明询问谁。

### Day 12：只读连接

目标：连接设备但不发送运动命令。

- [ ] 在断电状态下确认线缆和电源，不自行改变电压或总线参数。
- [ ] 使用官方 SDK/实验室已有脚本读取状态。
- [ ] 记录所有关节位置、速度、错误码、时间戳和采样频率。
- [ ] 连续读取 5–10 分钟，统计丢包、延迟和时间戳是否单调。
- [ ] 断开再连接，确认不会自动使能或突然运动。

通过标准：你能解释每个返回字段，且重连失败时机器人保持安全状态。

### Day 13：理解控制模式和限位

在负责人陪同下确认：

- 位置控制、速度控制、力矩控制的区别；
- SDK 的角度单位是度还是弧度；
- 发送的是目标位置还是增量；
- 速度和加速度限制在哪里生效；
- 软件停止、硬件急停和断电的区别。

不要为了“试试看”关闭限位、碰撞检测或急停。

### Day 14：单关节空载低速

采用“一个关节、一个小动作、一次只改一个参数”的原则：

1. 由负责人摆到安全姿态。
2. 选择一个关节，记录初始反馈。
3. 发送文档允许的最小动作。
4. 观察方向、幅度、延迟、停止响应和反馈。
5. 回到初始姿态，保存命令和结果。

停止条件：方向错、运动范围异常、抖动、噪声、过热、停止后继续运动、碰撞检测无效。

### Day 15：全臂遥操作，不录数据

- [ ] 先完成“回安全姿态—移动—停止—恢复”。
- [ ] 逐关节验证正负方向和夹具开合。
- [ ] 连续操作 10–20 分钟，观察温度、延迟、丢包和错误码。
- [ ] 测试急停、软件停止、通信中断三种情况。
- [ ] 写出从异常发生到恢复运行的具体步骤。

通过标准：你能在不使用 policy 的情况下稳定完成一次空载移动，并能明确什么时候必须按急停。

---

## 第 4 周：相机、标定和遥操作数据

### Day 16：相机入门和固定

官网：[Cameras](https://huggingface.co/docs/lerobot/main/cameras)。

现场确认：

- 相机设备索引/序列号；
- 分辨率、FPS、曝光、白平衡；
- 相机是否固定，支架是否会被碰；
- 画面是否同时看到物体、夹具和接触过程；
- 是否有反光、遮挡、运动模糊和过曝。

不要在正式数据中途改变相机位置。相机位置、焦距、背景和光照都是数据的一部分。

### Day 17：标定

PiperX 的标定可能包含关节零位、工具坐标、夹具行程、相机外参或 SDK 内部参数。按松灵文档执行，不要使用 SO-101 的 `lerobot-calibrate` 命令代替。

- [ ] 备份标定文件并记录路径。
- [ ] 记录每个关节的零位和允许范围。
- [ ] 记录末端工具坐标、夹具开合范围。
- [ ] 标定后用只读和低速测试确认姿态没有突然偏移。

### Day 18：录制前练习

官网：[Record a dataset](https://huggingface.co/docs/lerobot/main/il_robots#record-a-dataset)。

只做 3 次不保存练习：

1. 回到相同起始姿态。
2. 接近方块。
3. 闭合夹具并确认抓住。
4. 移到容器上方。
5. 打开夹具并回到安全姿态。

记录完成任务需要多少秒、哪些动作容易犹豫、相机是否能看清关键阶段。

### Day 19：三个测试 episode

- [ ] 录 3 个 episode。
- [ ] 立即用 `lerobot-dataset-viz` 或官网可视化工具检查。
- [ ] replay 前确认机器人在安全姿态、速度低、急停有人看守。
- [ ] 检查视频、state、action、timestamp、task 是否都有。

发现黑帧、丢帧、动作错位、状态维度错或 replay 不安全时，停止录制并修复，不要继续凑数量。

### Day 20：正式采集

录 30–50 个高质量 episode。前 10 个后检查一次，之后每 10 个检查一次。每个 episode 约 20–30 秒，保持相同任务描述和动作节奏，但覆盖少量合理变化：物体位置、起始姿态、接近角度。

训练前门槛：

- 每个 episode 有完整图像、状态、动作、时间戳和任务文本；
- 没有越过安全动作范围；
- 明显失败样本单独标记；
- 数据能加载、可视化和 replay；
- 能说清 state/action 的关节顺序和单位。

---

## 第 5 周：用 PiperX 数据训练 ACT

### Day 21：数据审计

官网：[Using Dataset Tools](https://huggingface.co/docs/lerobot/main/using_dataset_tools)。

逐项检查：

- episode 数和总帧数；
- 相机 FPS 与机器人状态 FPS；
- 图像分辨率和 dtype；
- action 的最小值、最大值、均值和异常尖峰；
- 每个 episode 的长度；
- 任务文字是否一致；
- 是否存在跨 episode 的时间窗口。

产出 `experiments/piperx-v1/data-audit.md` 和一张数据质量表。

### Day 22：smoke test

先用极短训练确认：

- policy 输入 feature 与 PiperX 数据一致；
- action 输出维度正确；
- normalization 能加载；
- loss 是有限数；
- checkpoint 可以重新加载。

smoke test 失败时不要直接增加显存或 steps，先检查 key、shape、dtype、单位、关节顺序和 processor。

### Day 23–24：正式训练

- [ ] 保存完整配置和实际命令。
- [ ] 保存至少两个 checkpoint。
- [ ] 记录训练步数、耗时、显存、loss 和数据版本。
- [ ] 固定一个验证集或固定若干 episode 做离线比较。

不要用训练 loss 直接宣称“机器人学会了”。离线模型能复现示范动作，不代表它能处理相机延迟、初始位置变化和真实摩擦。

### Day 25：离线动作检查

在不连接机械臂的情况下：

- 输入录制视频和状态；
- 运行 `select_action()`；
- 画出每个关节的预测曲线；
- 检查是否有尖峰、跳变、超限和不合理夹具动作；
- 比较示范动作与预测动作的时间对齐。

通过标准：没有任何动作会在数值上越过已确认的 PiperX 安全范围。

---

## 第 6 周：分阶段部署到 PiperX

部署前必须两人到场：一人操作电脑，一人站在急停旁。任何一个检查不通过都停止。

### 阶段 A：模型加载，不连接动作

确认 checkpoint、processor、数据统计、模型输出维度和 PiperX 控制接口一致。只打印动作，不发送。

### 阶段 B：影子模式

读取实时图像和状态，运行模型并记录预测动作，机器人保持不动。检查实时输入是否与训练数据同分布，推理 FPS 是否稳定。

### 阶段 C：空载低速

不放方块，只在安全姿态附近执行极短动作。设置速度、加速度和动作幅度上限。观察方向、延迟、抖动、停止响应和异常恢复。

### 阶段 D：单次有物体测试

只放一个方块，完成一次后停止检查日志。失败也要保存视频和状态，不要连续重试掩盖问题。

### 阶段 E：固定条件评估

完成 10 个 episode，记录：

| 字段 | 说明 |
| --- | --- |
| episode_id | 唯一编号 |
| 初始姿态/物体位置 | 评估条件 |
| 成功 | 0/1，预先定义成功标准 |
| 失败阶段 | 接近、抓取、搬运、放置 |
| 人工干预/急停 | 是否发生、原因 |
| 推理 FPS/通信错误 | 系统稳定性 |
| 视频和日志路径 | 可复查证据 |

成功标准必须在实验前写好，例如“方块完全进入容器且夹具回到安全姿态”。不要看到结果后再改标准。

---

## 第 7 周：失败分析和第二版数据

按失败阶段决定下一步：

- **接近失败**：检查相机视角、初始位置覆盖、坐标/单位和动作平滑性。
- **抓取失败**：补录接近角度、夹具开合和接触过程；检查夹具状态是否进入 observation。
- **搬运抖动**：检查控制频率、action chunk、时间戳、增量/绝对动作表示。
- **放置失败**：补录容器边界和最后放置动作；检查目标是否超出训练分布。
- **通信/急停异常**：停止模型迭代，修复控制链路后重新完成第 3 周 Level 1–3。

只改一个主要变量。补录 10–20 个针对性 episode，训练第二版 policy，用完全相同的 10-episode 条件对比第一版和第二版。

最终报告至少包含：

1. 硬件和软件版本；
2. 数据集组成和质量检查；
3. policy、batch、steps、checkpoint；
4. 两版成功率；
5. 每类失败的数量和视频索引；
6. 下一轮最值得改的一件事。

---

## 第 8 周：整理成可复现项目

- [ ] 把所有命令整理成 `experiments/*/command.txt`。
- [ ] 把配置、数据集 repo、checkpoint 路径和 commit 写入实验表。
- [ ] 补齐 `notes/` 中没有回答的问题。
- [ ] 画最终数据流：相机/状态 → processor → policy → action → PiperX。
- [ ] 写一页“如果换一台电脑，如何从零复现实验”。
- [ ] 只提交代码、配置、日志摘要和获许可的小图，不提交 token、完整视频、HF cache、checkpoint 或未经许可的实验室图像。

---

## 每日学习记录模板

```markdown
# YYYY-MM-DD / Day XX

## 今天阅读
- 官网页面：
- 章节：
- 新术语：

## 今天执行
- 命令/脚本：
- 输入：
- 输出路径：

## 我现在能解释
-

## 观察到的证据
-

## 仍然不懂或不确定
-

## 明天只做的一件事
-
```

## 每次实机操作卡

开始前：检查场地、线缆、夹具、急停、速度限制、人员位置、模型版本和当前错误码。

运行中：只做当前阶段允许的动作；出现异响、过热、抖动、失联、超限或人进入危险区，立即停止。

结束后：回安全姿态，按负责人规定停止/失能/断电；保存日志、视频、状态、动作和备注；立即写下“现象—推测—证据—下一步”。

## 最终自测问题

完成后不看文档回答：

1. observation、state、action、task、episode 分别是什么？
2. 为什么 episode 边界不能跨越？
3. 相机 FPS、机器人状态频率和 policy 推理频率有什么区别？
4. ACT 的 action chunk 解决什么问题，又引入什么要求？
5. `forward()` 和 `select_action()` 有什么区别？
6. 为什么训练 loss 下降不代表实机成功率上升？
7. PiperX 的一个 action 数值具体代表什么单位和控制目标？
8. 影子模式为什么要放在真正发送动作之前？
9. 发生抖动、失联或急停后，分别如何停止和恢复？
10. 如果抓取阶段总失败，下一轮应优先检查相机、数据、动作表示还是模型？为什么？

能够回答这些问题，并完成 10 次固定条件实机评估，才算完成第一次 LeRobot 项目。此后再学习 SmolVLA、Diffusion Policy、RL 或把 PiperX 做成正式的 LeRobot 原生适配。
