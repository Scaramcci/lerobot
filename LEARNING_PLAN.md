# LeRobot 三周学习计划：从公开数据到实机闭环

> 适用对象：具身智能初学者，读过少量论文，但没有机器人项目经验  
> 当前条件：Ubuntu 26.04、RTX 4090 Laptop、工作日每天 2–3 小时、周末每天 4–6 小时  
> 实机条件：实验室有机械臂，预计一至两周后可以使用，具体型号待确认  
> 官网版本：LeRobot `main` 文档  
> 目标周期：3 周完成，必要时使用第 4 周缓冲

## 1. 最终目标

本计划的目标不是做一个复杂的“简历项目”，而是完整经历一次：

```text
安装 → 理解数据 → 训练 Policy → 仿真评估
    → 连接机械臂 → 标定与遥操作 → 采集数据
    → 训练实机 Policy → 部署 → 定量评估
```

完成后，应当能够：

- [ ] 解释 observation、state、action、episode、task、FPS 的含义。
- [ ] 说明 LeRobotDataset 如何保存视频、机器人状态和动作。
- [ ] 使用公开数据训练并评估一个 ACT Policy。
- [ ] 理解 action chunk，以及训练和推理阶段的基本数据流。
- [ ] 完成一种机械臂的连接、标定、遥操作和数据录制。
- [ ] 用自己录制的数据训练 ACT，并完成至少 10 次实机测试。
- [ ] 使用成功率、失败阶段和数据质量分析结果，而不是只看训练 loss。
- [ ] 把配置、命令、实验结果和学习笔记提交到个人 GitHub。

## 2. 官网学习范围

官网入口：[LeRobot Documentation](https://huggingface.co/docs/lerobot/main/index)

### 2.1 必学内容

| 顺序 | 官网位置 | 页面 | 学习深度 |
|---:|---|---|---|
| 1 | Get started → LeRobot | [LeRobot 首页](https://huggingface.co/docs/lerobot/main/index) | 精读 How It Works 和 Get Started |
| 2 | Get started → Installation | [Installation](https://huggingface.co/docs/lerobot/main/installation) | 精读并完成源码安装、FFmpeg、CUDA 验证 |
| 3 | Get started → Cheat sheet | [Cheat sheet](https://huggingface.co/docs/lerobot/main/cheat-sheet) | 浏览，作为命令索引，不背命令 |
| 4 | Compute & Hardware → Compute Hardware Guide | [Compute Hardware Guide](https://huggingface.co/docs/lerobot/main/hardware_guide) | 精读显存、batch size、训练时长和 checkpoint 部分 |
| 5 | Datasets → Using LeRobotDataset | [LeRobotDataset v3](https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3) | 精读格式、加载、索引、时间窗口和 DataLoader |
| 6 | Datasets → Using the Dataset Tools | [Dataset Tools](https://huggingface.co/docs/lerobot/main/using_dataset_tools) | 学会查看信息和可视化；其他操作先浏览 |
| 7 | Policies → ACT | [ACT](https://huggingface.co/docs/lerobot/main/act) | 精读架构、训练和评估 |
| 8 | Tutorials → Imitation Learning for Robots | [Imitation Learning for Robots](https://huggingface.co/docs/lerobot/main/il_robots) | 全文精读；这是实机阶段的主教程 |
| 9 | Sensors → Cameras | [Cameras](https://huggingface.co/docs/lerobot/main/cameras) | 实机前精读找相机、配置和图像读取 |
| 10 | Inference → Policy Deployment | [Policy Deployment](https://huggingface.co/docs/lerobot/main/inference) | 先学 Quick Start、Base strategy 和 Common Flags |
| 11 | Robot Processors → Introduction | [Introduction to Processors](https://huggingface.co/docs/lerobot/main/introduction_processors) | 完成第一次训练后再读，理解数据流即可 |
| 12 | Robot Processors → Action Representations | [Action Representations](https://huggingface.co/docs/lerobot/main/action_representations) | 理解 joint/EE、absolute/relative/delta 的区别 |
| 13 | Robots → 对应机械臂 | [LeRobot 文档首页](https://huggingface.co/docs/lerobot/main/index)左侧的 Robots 栏目 | 只学习实验室实际使用的型号 |

### 2.2 选学内容

以下内容只有在主线任务完成或遇到对应需求时才学习：

- [PyTorch Accelerators](https://huggingface.co/docs/lerobot/main/torch_accelerators)：需要排查 CUDA/device 问题时阅读。
- [Notebooks](https://huggingface.co/docs/lerobot/main/notebooks)：本机环境失败时作为备用，不作为主线。
- [SmolVLA](https://huggingface.co/docs/lerobot/main/smolvla)：完成 ACT 实机闭环后，用第 4 周了解 VLA。
- [Debug Processor Pipeline](https://huggingface.co/docs/lerobot/main/debug_processor_pipeline)：遇到 shape、key 或归一化问题时阅读。
- [Third-Party Robots](https://huggingface.co/docs/lerobot/main/third_party_robots)：实验室机械臂不在官方 Robots 列表时查阅。
- [LeLab](https://huggingface.co/docs/lerobot/main/lelab)：如果使用 SO-101，可在掌握 CLI 后体验 GUI。

### 2.3 本轮明确不学

这些内容有价值，但会明显分散第一次端到端学习的注意力：

- Adding a Policy、Bring Your Own Hardware：本轮不实现新 Policy，也不接入新机器人驱动。
- RL、HIL-SERL、Human in the Loop：先建立监督式模仿学习基线。
- Multi-GPU、PEFT：单张 4090 Laptop 足够完成本计划。
- π0、π0.5、π0-FAST、GR00T、X-VLA、MolmoAct2 等大型 VLA：先完成 ACT 闭环。
- Reward Models：本轮不训练奖励模型。
- Async Inference、RTC、Sentry、DAgger 等高级推理策略：第一版使用同步 Base strategy。
- LIBERO、Meta-World、RoboCasa、RoboTwin 等大型 benchmark：本轮只用 PushT 做最小仿真闭环。
- Porting Large Datasets、Annotation Pipeline、Video Encoding 高级参数：只有出现实际需求再学。
- Implement Your Own Processor、Environment Processor：本轮只理解并使用现有 Processor。
- API Reference：只作为查阅工具，不按顺序通读。
- Contributing、Writing Docstrings、Backward Compatibility：等需要向上游提交代码时再学习。

## 3. 第 1 周：公开数据与仿真闭环

目标：不依赖实验室，在本机完成安装、数据理解、ACT 训练和 PushT 仿真评估。

### Day 1：认识流程和完成安装（2–3 小时）

官网：

1. Get started → [LeRobot 首页](https://huggingface.co/docs/lerobot/main/index)
2. Get started → [Installation](https://huggingface.co/docs/lerobot/main/installation)
3. Compute & Hardware → [Compute Hardware Guide](https://huggingface.co/docs/lerobot/main/hardware_guide)

任务：

- [ ] 用一句话解释 Teleoperate → Record → Train → Deploy。
- [ ] 按官网源码安装方式建立隔离环境。
- [ ] 安装 FFmpeg，并确认 `ffmpeg -version` 正常。
- [ ] 确认 PyTorch 能识别 RTX 4090 Laptop。
- [ ] 记录 Python、PyTorch、CUDA、NVIDIA Driver 和 LeRobot commit。
- [ ] 运行一个小范围测试或 CLI `--help`，确认基本安装可用。

验收：能够在项目环境中导入 `torch` 和 `lerobot`，并看到 CUDA 设备名称。

### Day 2：建立仓库地图（2–3 小时）

官网：Get started → [Cheat sheet](https://huggingface.co/docs/lerobot/main/cheat-sheet)

只认识以下目录，不逐文件通读：

- `src/lerobot/scripts/`：CLI 入口。
- `src/lerobot/configs/`：配置和命令行参数。
- `src/lerobot/datasets/`：数据集。
- `src/lerobot/policies/`：ACT 等 Policy。
- `src/lerobot/processor/`：模型前后处理。
- `src/lerobot/envs/`：仿真环境。
- `src/lerobot/robots/`、`cameras/`、`teleoperators/`：实机接口。

任务：

- [ ] 找到 `lerobot-train`、`lerobot-eval`、`lerobot-record` 的源码入口。
- [ ] 找到 ACT 的配置类、Policy 类和注册位置。
- [ ] 画一张不超过十个节点的仓库结构图。

验收：可以说明“一条训练命令从 CLI 进入后，大致会经过哪些模块”。

### Day 3：理解 LeRobotDataset（2–3 小时）

官网：Datasets → [LeRobotDataset v3](https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3)

练习数据：[lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht)

重点只学：

- Format design 和目录结构。
- 从 Hub 加载数据。
- Random access by index。
- `delta_timestamps` 时间窗口。
- DataLoader。

任务：

- [ ] 在线查看 PushT 的一个 episode。
- [ ] 本地加载 metadata 和一个 sample。
- [ ] 写下 sample 中每个 key 的名称、shape、dtype 和含义。
- [ ] 解释为什么 episode 边界不能当成普通连续帧跨越。
- [ ] 解释 observation 与 action 的时间对应关系。

验收：能够拿一个 batch，说明 batch 中每个张量代表什么。

### Day 4：可视化和数据质量（2–3 小时）

官网：Datasets → [Dataset Tools](https://huggingface.co/docs/lerobot/main/using_dataset_tools)

本轮只学习：

- Show dataset information。
- Online Visualization。
- Local Visualization。
- 知道 delete、split、merge 存在，但先不修改官方数据。

任务：

- [ ] 使用官方 Dataset Visualizer 查看 PushT。
- [ ] 使用 `lerobot-dataset-viz` 本地查看 episode 0。
- [ ] 检查帧是否连续、动作是否合理、成功标记如何变化。
- [ ] 写一页“训练前数据检查清单”。

验收：能够说出至少五个需要在训练前检查的数据质量问题。

### Day 5：学习 ACT（2–3 小时）

官网：Policies → [ACT](https://huggingface.co/docs/lerobot/main/act)

需要理解：

- 图像 backbone、Transformer encoder/decoder 的职责。
- 什么是 action chunk。
- 训练时的输入、监督信号和 loss。
- 推理时为什么不一定每帧都重新规划完整轨迹。
- ACT 为什么适合作为第一个实机 Policy。

不要求：手推完整 Transformer 公式或复现论文代码。

验收：不用看文档，可以用自己的话讲清 ACT 的输入、输出和 action chunk。

### Weekend：训练并评估 PushT（8–12 小时）

官网：

- Policies → [ACT](https://huggingface.co/docs/lerobot/main/act)
- Tutorials → [Imitation Learning for Robots：Train a policy](https://huggingface.co/docs/lerobot/main/il_robots#train-a-policy)

任务：

- [ ] 先运行约 200 steps 的 smoke test。
- [ ] 再完成一次约 5,000 steps 的正式 ACT 训练。
- [ ] 保存至少两个中间 checkpoint。
- [ ] 在 PushT 环境快速评估 20 个 episode。
- [ ] 最终评估 50 个 episode。
- [ ] 记录训练时长、峰值显存、loss 和成功率。
- [ ] 至少改变一个变量做对照，例如训练步数或 batch size。

验收：得到可加载的 checkpoint、评估视频或日志，以及一张两组实验的对比表。

## 4. 第 2 周：从训练脚本走向完整实机流程

目标：理解训练/推理数据流，并在进入实验室前把实机教程读完。

### Day 6：追踪训练数据流（2–3 小时）

按照下面一条路径阅读代码：

```text
LeRobotDataset → DataLoader → Policy Processor
→ ACT.forward() → loss → optimizer → checkpoint
```

任务：

- [ ] 找到 dataset metadata 如何转成 policy features。
- [ ] 找到 normalization 在哪里发生。
- [ ] 找到 checkpoint 中保存了哪些文件。
- [ ] 区分 config、model weights、processor state 和 dataset stats。

验收：能够从训练入口定位到 `ACTPolicy.forward()`。

### Day 7：追踪推理数据流（2–3 小时）

官网：Inference → [Policy Deployment](https://huggingface.co/docs/lerobot/main/inference)

本轮只学：

- Quick Start。
- Base strategy。
- Sync backend。
- Common Flags 和 cadence report。

路径：

```text
Camera/Robot observation → preprocessor → select_action()
→ postprocessor → robot.send_action()
```

验收：能够解释训练阶段的 `forward()` 与部署阶段的 `select_action()` 有什么不同。

### Day 8：理解 Processor（2–3 小时）

官网：Robot Processors → [Introduction to Processors](https://huggingface.co/docs/lerobot/main/introduction_processors)

只掌握：

- ProcessorStep。
- RobotProcessorPipeline 与 PolicyProcessorPipeline。
- observation/action feature contract。
- normalize、batch、device、rename 等常见 step 的作用。

验收：能画出从相机图像到模型、再到关节动作的前后处理位置。

### Day 9：动作表示（2–3 小时）

官网：Robot Processors → [Action Representations](https://huggingface.co/docs/lerobot/main/action_representations)

任务：

- [ ] 区分 joint space 与 end-effector space。
- [ ] 区分 absolute、relative 和 delta action。
- [ ] 写出实验室机械臂预计使用的状态与动作表示；不确定时标为待确认。
- [ ] 说明训练数据和部署时动作表示为什么必须一致。

### Day 10：通读实机主教程（2–3 小时）

官网：Tutorials → [Imitation Learning for Robots](https://huggingface.co/docs/lerobot/main/il_robots)

按以下顺序整理自己的命令模板，但不要在未知硬件上执行：

1. Set up and Calibrate。
2. Teleoperate。
3. Cameras。
4. Record a dataset。
5. Visualize a dataset。
6. Replay an episode。
7. Train a policy。
8. Run inference and evaluate。

验收：形成一张“实机当天操作清单”，每一步都有停止条件和预期输出。

### Weekend：ALOHA 迁移练习 + 实机准备（8–12 小时）

练习数据：[lerobot/aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human)

任务：

- [ ] 可视化至少一个双臂 episode。
- [ ] 比较 ALOHA 与 PushT 的 observation/action shape、FPS、相机和任务难度。
- [ ] 用 ACT 完成一次训练 smoke test；时间足够再做正式训练。
- [ ] 确认实验室机械臂准确型号。
- [ ] 确认 teleoperator、相机、接口、电源、急停和场地权限。
- [ ] 在官网 Robots 栏目找到对应硬件页，并将链接填入下一节。

不要求：本周必须把 ALOHA 训练到高成功率。它只是用于验证代码能迁移到不同机器人形态的数据。

## 5. 第 3 周：实机闭环

目标：完成一项极简任务的数据采集、ACT 训练、部署和评估。

### 实验室硬件信息

- 机械臂型号：`待确认`
- 官网对应位置：Robots → `待确认`
- 官网链接：`待确认`
- Teleoperator：`待确认`
- 相机：`待确认`
- 控制计算机：`待确认`
- 紧急停止方式：`待确认`

如果官网没有该型号，查看 [Third-Party Robots & Teleoperators](https://huggingface.co/docs/lerobot/main/third_party_robots)。在确认型号、电机类型、电压和通信接口前，不执行电机配置命令。

### Session 1：连接、标定和遥操作（3–5 小时）

官网：

- Robots → 对应机械臂页面。
- Tutorials → [Set up and Calibrate](https://huggingface.co/docs/lerobot/main/il_robots#set-up-and-calibrate)
- Tutorials → [Teleoperate](https://huggingface.co/docs/lerobot/main/il_robots#teleoperate)

任务：

- [ ] 确认电源、电压、线缆、关节限位和急停。
- [ ] 找到所有串口、相机或网络设备。
- [ ] 完成标定并备份标定文件。
- [ ] 低速遥操作，验证每个关节方向和 gripper。
- [ ] 运行 10–20 分钟后检查通信稳定性、温度和延迟。

停止条件：关节方向错误、异常噪声、过热、失联、相机明显延迟或急停不可用。

### Session 2：相机和数据采集（4–6 小时）

官网：

- Sensors → [Cameras](https://huggingface.co/docs/lerobot/main/cameras)
- Tutorials → [Record a dataset](https://huggingface.co/docs/lerobot/main/il_robots#record-a-dataset)
- Tutorials → [Visualize a dataset](https://huggingface.co/docs/lerobot/main/il_robots#visualize-a-dataset)
- Tutorials → [Replay an episode](https://huggingface.co/docs/lerobot/main/il_robots#replay-an-episode)

第一项任务建议：固定位置抓起单个方块，并放入固定容器。

约束：

- 单物体、固定背景、固定相机、稳定光照。
- 每个 episode 约 20–30 秒。
- 先练习 5–10 次，再正式录制。
- 第一版录制 30–50 个高质量 episode。
- 动作策略保持一致；明显失败或犹豫的 episode 立即重录。

任务：

- [ ] 检查相机画面能否独立支持完成任务。
- [ ] 录制 3 个测试 episode，立即可视化和 replay。
- [ ] 测试无误后录制正式数据。
- [ ] 训练前逐 episode 抽查视频、state、action、时间戳和任务文本。
- [ ] 记录数据集 repo ID、任务描述、episode 数、FPS、相机位置。

验收：得到能够加载、可视化和 replay 的 LeRobotDataset。

### Session 3：训练、部署和评估（4–6 小时）

官网：

- Policies → [ACT](https://huggingface.co/docs/lerobot/main/act)
- Tutorials → [Train a policy](https://huggingface.co/docs/lerobot/main/il_robots#train-a-policy)
- Inference → [Policy Deployment](https://huggingface.co/docs/lerobot/main/inference)
- Tutorials → [Run inference and evaluate](https://huggingface.co/docs/lerobot/main/il_robots#run-inference-and-evaluate-your-policy)

任务：

- [ ] 先完成短 smoke test，验证 features 和 shape 匹配。
- [ ] 根据总帧数和 batch size 估算 5–10 epochs 所需 steps。
- [ ] 训练 ACT，并保存多个 checkpoint。
- [ ] 部署前在无障碍位置低速测试动作方向和范围。
- [ ] 固定初始条件完成至少 10 次评估。
- [ ] 记录成功率，并标记每次失败发生在哪个阶段。
- [ ] 如果失败集中在同一阶段，补录 10–20 个针对性 episode，再训练一次。

验收：得到第一版和改进版结果，并能说明性能变化主要来自数据、训练还是部署设置。

## 6. 第 4 周缓冲与选修

只有以下情况才启用第 4 周：

- 实验室审批或排期延迟。
- 机械臂不受 LeRobot 原生支持。
- 相机、串口、标定或数据格式需要额外排查。
- 第一版数据质量不足，需要重新采集。

优先级：

1. 完成尚未结束的实机闭环。
2. 补做针对性数据采集和 ACT 重训。
3. 完成 ACT 与 Diffusion Policy 的 PushT 对照实验。
4. 主线全部完成后，再阅读 [SmolVLA](https://huggingface.co/docs/lerobot/main/smolvla)，只做概念了解或 smoke test。

第 4 周仍不开始 RL、新 Policy 实现或大型 benchmark。

## 7. 每日学习记录模板

每次学习结束后在 `notes/` 中保存一份简短记录：

```markdown
# YYYY-MM-DD

## 今天阅读
- 官网栏目：
- 页面：
- 读到的章节：

## 今天执行
- 命令或代码：
- 输入数据：
- 输出位置：

## 结果
- 耗时：
- GPU 峰值显存：
- loss / 成功率：
- 生成的 checkpoint：

## 我能解释
- 

## 尚未理解
- 

## 下一步
- 
```

## 8. 实验记录模板

| 实验 ID | 数据集 | Policy | Batch | Steps / Epochs | 训练耗时 | 峰值显存 | Eval episodes | 成功率 | 唯一改变量 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| exp-001 | lerobot/pusht | ACT |  |  |  |  |  |  | baseline |
| exp-002 | lerobot/pusht | ACT |  |  |  |  |  |  |  |
| exp-003 | 自采数据 | ACT |  |  |  |  | 10 |  | baseline |

## 9. GitHub 中应保存与不应保存的内容

应该保存：

- 本学习计划。
- `notes/` 学习笔记。
- 可复现实验配置和实际执行命令。
- 小型分析脚本。
- 实验结果表和少量压缩后的示例图片。
- 失败原因与解决过程。

不应保存：

- `outputs/` 中的 checkpoint。
- Hugging Face 数据集缓存和完整视频数据。
- `wandb/` 日志目录。
- Token、密码、`.env`。
- 未经实验室许可的图像、数据和设备信息。

数据集和模型应上传到 Hugging Face Hub；GitHub 只保存代码、配置、结果摘要和学习记录。

## 10. 最终复盘问题

完成计划后，不看文档回答：

1. LeRobotDataset 为什么以 episode 为核心，而不是普通独立图片？
2. observation、state、action、task 和 timestamp 如何对齐？
3. ACT 为什么一次预测多个未来动作？
4. Processor 为什么需要同时服务训练和实机推理？
5. loss 下降为什么不等于实机成功率提高？
6. 数据质量、相机位置和动作一致性分别会造成什么失败？
7. 仿真到实机增加了哪些问题？
8. 如何设计一次可复现的 10-episode 实机评估？
9. 如果 Policy 总在抓取阶段失败，下一轮应该优先改什么？
10. 下一阶段应该学习 SmolVLA、RL，还是继续改进数据？为什么？

能够清楚回答这些问题，并完成一次真实机器人定量评估，即视为本轮学习完成。
