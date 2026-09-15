# `src/lerobot` 源码结构说明

本文档对应当前仓库中的 `src/lerobot`。它是 LeRobot 的 Python 包源码：上层脚本负责把命令行参数转换为配置并编排流程，中间层负责数据集、处理器、策略和训练/推理，下层负责机器人、遥操作器、相机和电机硬件。

## 1. 总体架构

```text
scripts/  (lerobot-train / record / eval / ...)
    |
configs/ ---- datasets/ ---- processor/ ---- policies/ ---- optim/
    |             |              |               |
envs/       storage/video    observation     model + processor
    |                                             |
robots/ <---- teleoperators/ ---- cameras/ ---- motors/
    |
rollout/ / async_inference/ / transport/       utils/ common/
```

典型训练链路是“配置 -> 数据集 -> 预处理器 -> 策略模型 -> 优化器/分布式训练”；典型部署链路是“相机和机器人读取观测 -> 策略推理 -> 后处理 -> 机器人执行”。`lerobot_types.py` 中的 `RobotObservation`、`RobotAction`、`PolicyAction`、`EnvTransition` 是这些层之间共享的数据契约。

## 2. 顶层文件

| 文件 | 作用 |
|---|---|
| `__init__.py` | 包入口，只导出版本并说明可选依赖分组。 |
| `__version__.py` | 定义包版本号。 |
| `lerobot_types.py` | 统一的 TypedDict、枚举和张量/数组类型；定义观测、动作、奖励、环境转移等键。 |

## 3. 配置与基础类型：`configs/`

配置使用 dataclass、`draccus.ChoiceRegistry` 和 YAML/CLI 解析；策略、环境、机器人和数据集都通过配置注册表实现多态选择。

| 文件 | 作用 |
|---|---|
| `__init__.py` | 汇总公开配置类型，避免导出会造成循环依赖的训练配置。 |
| `accelerator.py` | Accelerate/设备、混合精度和梯度累积配置。 |
| `dataset.py` | 数据集记录、数据集来源和采样相关配置。 |
| `default.py` | 通用作业、评估、EMA、PEFT、W&B 等默认配置。 |
| `eval.py` | 评估流水线配置。 |
| `parallelism.py` | 张量/数据并行等并行训练配置。 |
| `parser.py` | 将 dataclass 配置解析为 CLI、YAML，并处理选择注册表。 |
| `policies.py` | `PreTrainedConfig` 基类及策略公共字段（输入特征、设备、归一化等）。 |
| `recipe.py` | 训练 recipe、消息轮次和 recipe 文件加载。 |
| `rewards.py` | 奖励模型配置基类和公共参数。 |
| `train.py` | `TrainPipelineConfig`，训练任务的顶层配置组合。 |
| `types.py` | 特征类型、归一化模式、策略特征和 RTC 调度等类型。 |
| `video.py` | RGB/深度视频编码器及视频格式配置。 |
| `recipes/subtask*.yaml` | 子任务标注/记忆/VQA-语音等训练 recipe 示例。 |

## 4. 数据集：`datasets/`

此目录实现 LeRobot 数据集格式（Parquet 元数据/帧索引、图像或视频、episode/task 划分），并提供本地、Hub、流式和 Lance 后端。

| 文件 | 作用 |
|---|---|
| `__init__.py` | 检查 `datasets`/`av` 依赖并导出公开 API。 |
| `lerobot_dataset.py` | `LeRobotDataset` 主数据集类：按 episode 采样、读取帧和训练样本。 |
| `dataset_metadata.py` | 读取/写入 `meta/info.json`、统计量、特征和 episode/task 元数据。 |
| `dataset_reader.py` | 数据集读取器抽象及 Parquet/远程读取实现。 |
| `dataset_writer.py` | 将帧、episode 和元数据写入 LeRobot 格式。 |
| `factory.py` | 根据配置创建普通、多数据集或流式数据集，并构建采样器。 |
| `multi_dataset.py` | 合并多个数据集并协调特征、任务和采样权重。 |
| `streaming_dataset.py` | 面向远程/大数据的流式迭代数据集。 |
| `sampler.py` | episode-aware、权重和分布式采样器。 |
| `aggregate.py` | 跨数据集聚合统计量。 |
| `compute_stats.py` | 计算均值、方差、分位数等特征统计量。 |
| `feature_utils.py` | 校验、推断和转换数据集特征描述。 |
| `pipeline_features.py` | 将处理流水线的输入/输出特征写入数据集描述。 |
| `dataset_tools.py` | 删除/合并 episode、修改 task/feature、重编码和重算统计量等工具。 |
| `image_writer.py` | 异步写入图像帧。 |
| `video_utils.py` | 视频编码、解码和帧索引辅助。 |
| `pyav_utils.py` | PyAV 解码器封装。 |
| `depth_utils.py` | 深度图单位、编码和转换。 |
| `language.py` | 语言任务字段的标准化。 |
| `language_render.py` | 将语言任务渲染为可视化/提示文本。 |
| `io_utils.py` | 数据集文件系统和安全 IO 辅助。 |
| `storage.py` | 本地、Hub、S3 等存储 URI 和后端选择。 |
| `lance_backend.py` / `lance_utils.py` | Lance 格式读写及其工具函数。 |
| `utils.py` | episode 索引、路径和数据集通用辅助。 |
| `card_template.md` | 数据集 Hub model card 模板。 |

## 5. 处理流水线：`processor/`

`pipeline.py` 提供可注册的 `ProcessorStep` 和串行 `DataProcessorPipeline`/`PolicyProcessorPipeline`。一个策略通常有“数据集 -> 模型”的 preprocessor 和“模型输出 -> 机器人”的 postprocessor。

| 文件 | 作用 |
|---|---|
| `pipeline.py` | 流水线、步骤注册、配置序列化和 Hub 保存/加载。 |
| `factory.py` | 根据策略配置构建默认 pre/post processor。 |
| `converters.py` | observation/action/transition 与 numpy/tensor 之间的转换。 |
| `batch_processor.py` | 添加 batch 维度。 |
| `device_processor.py` | 将张量移动到 CPU/CUDA/MPS 等设备。 |
| `observation_processor.py` | 观测字段映射与抽象观测处理步骤。 |
| `env_processor.py` | 环境观测/动作与策略格式之间的桥接。 |
| `gym_action_processor.py` | Gym action space 格式转换。 |
| `normalize_processor.py` | 按数据集统计量归一化/反归一化。 |
| `relative_action_processor.py` / `delta_action_processor.py` | 绝对动作、相对动作和 delta 动作互转。 |
| `rename_processor.py` | 重命名观测/动作字段。 |
| `device_processor.py` | 统一设备和 dtype。 |
| `tokenizer_processor.py` | 文本 tokenizer 和 token 字段处理。 |
| `newline_task_processor.py` | 为任务文本添加/规范换行。 |
| `render_messages_processor.py` | 将多模态消息渲染为模型输入。 |
| `hil_processor.py` | 人在回路（HIL）数据处理。 |
| `policy_robot_bridge.py` | 策略动作与机器人动作/观测的边界适配。 |
| `migrate_policy_normalization.py` | 迁移旧版策略归一化配置。 |

## 6. 策略模型：`policies/`

根目录的 `pretrained.py` 定义可保存到 Hugging Face Hub 的 `PreTrainedPolicy`；`factory.py` 延迟导入并按名称创建策略；`utils.py` 负责推理前后观测/动作整理。每个策略子目录通常遵循 `configuration_*`（配置）、`modeling_*`（PyTorch 模型）、`processor_*`（处理器）三件套。

| 文件/目录 | 作用 |
|---|---|
| `__init__.py` | 导出策略配置、工厂和公共工具。 |
| `pretrained.py` | 策略基类、权重保存/加载、Hub 集成和训练/推理接口。 |
| `factory.py` | 策略名称到类的映射、实例化和 pre/post processor 创建。 |
| `utils.py` | 推理输入准备、机器人动作生成等公共辅助。 |
| `pi_gemma.py` | PaliGemma/Gemma 视觉语言策略实现。 |
| `common/flow_matching.py` | 流匹配通用数学模块。 |
| `common/vla_utils.py` | VLA 模型共享的视觉/语言辅助。 |

以下目录均按同一约定实现一个具体策略；目录内的 `__init__.py` 注册/导出策略，`configuration_*.py` 定义超参数，`modeling_*.py` 定义网络和 loss，`processor_*.py` 定义输入输出处理：

`act/`（ACT，`configuration_act.py`、`modeling_act.py`、`processor_act.py`）；
`diffusion/`（Diffusion Policy，对应三个文件）；
`eo1/`（EO-1，对应三个文件）；
`evo1/`（Evo-1：另含 `evo1_model.py`、`flow_matching.py`、`internvl3_embedder.py`）；
`fastwam/`（FAST-WAM：三个主文件）；
`gaussian_actor/`（高斯策略：三个主文件）；
`groot/`（GR00T：`configuration_groot.py`、`modeling_groot.py`、`processor_groot.py`、`groot_n1_7.py`、`utils.py`，以及 `action_head/cross_attention_dit.py`）；
`lingbot_va/`（LingBot-VLA：三个主文件和 `utils.py`）；
`molmoact2/`（MolmoAct2：三个主文件、`utils_molmoact2.py`，以及 `molmoact2_hf_model/` 下的 action tokenizer、配置、图像/视频处理、推理和模型实现）；
`multi_task_dit/`（多任务 DiT：三个主文件）；
`pi0/`、`pi05/`、`pi0_fast/`（PI0、PI05、PI0-Fast，各含配置/模型/处理器；PI05 另含 `memory.py`）；
`smolvla/`（SmolVLA：三个主文件和 `smolvlm_with_expert.py`）；
`tdmpc/`（TD-MPC：三个主文件）；
`vla_jepa/`（VLA-JEPA：配置、模型、处理器、`action_head.py`、`qwen_interface.py`、`world_model.py`）；
`vqbet/`（VQ-BeT：配置、模型、处理器、`vqbet_utils.py`）；
`wall_x/`（Wall-X：配置、模型、处理器、常量和工具，`qwen_model/` 下为 Qwen2.5-VL MoE、视觉 attention 和配置）；
`xvla/`（XVLA：配置、模型、处理器、`action_hub.py`、`soft_transformer.py`、工具）。

`fastwam/wan/` 是 FAST-WAM 使用的 WAN 视频扩散组件：`adapters.py`（模型适配器）、`components.py`（组件）、`model.py`（模型）、`modular.py`（模块化组合）、`video_dit.py`（视频 DiT）和 `README.md`（组件说明）。

## 7. 环境、硬件与控制

### `envs/`

`configs.py` 定义环境配置基类；`factory.py` 创建 Gym 环境并把环境特征映射为策略特征；`libero.py`、`metaworld.py`、`robocasa.py`、`robomme.py`、`robotwin.py`、`vlabench.py` 分别接入对应仿真/基准环境；`utils.py` 提供环境包装和特征辅助；`metaworld_config.json` 保存 Meta-World 环境参数。

### `robots/`

`config.py` 是机器人配置基类，`robot.py` 是硬件生命周期/观测/动作抽象，`utils.py` 按配置创建机器人。其余目录是具体机器人适配器：

| 目录 | 文件职责 |
|---|---|
| `so_follower/` | SO-101 follower；配置、机器人实现和 `robot_kinematic_processor.py`。 |
| `bi_so_follower/` | 双臂 SO follower；配置和机器人实现。 |
| `koch_follower/` | Koch follower；配置和机器人实现。 |
| `omx_follower/` | OMX follower；配置和机器人实现。 |
| `openarm_follower/`、`bi_openarm_follower/` | OpenArm 单臂/双臂配置与实现。 |
| `rebot_b601_follower/`、`bi_rebot_b601_follower/` | Rebot B601 单臂/双臂配置与实现。 |
| `reachy2/` | Reachy 2 配置和机器人实现。 |
| `lekiwi/` | LeKiwi 配置、机器人、客户端和主机端服务。 |
| `earthrover_mini_plus/` | EarthRover Mini Plus 配置与实现。 |
| `hope_jr/` | Hope Jr 手臂和手部组件及配置。 |
| `unitree_g1/` | Unitree G1 配置、运动学、SDK socket、服务启动和控制器（GR00T/Holosoma/全身）。 |

### `cameras/`

`camera.py` 是相机抽象，`configs.py` 定义公共配置，`utils.py` 做设备发现/图像辅助。`opencv/`（OpenCV 摄像头）、`realsense/`（RealSense RGB-D）、`reachy2_camera/`（Reachy 2 相机）、`zmq/`（ZMQ 相机客户端和 `image_server.py`）各自提供配置和驱动实现。

### `motors/`

`motors_bus.py` 定义电机总线和校准接口，`encoding_utils.py` 处理寄存器/编码，`calibration_gui.py` 提供校准界面。`damiao/`、`dynamixel/`、`feetech/`、`robstride/` 各含 `<vendor>.py` 驱动和 `tables.py` 控制表；目录 `__init__.py` 导出驱动。

### `teleoperators/`

`teleoperator.py` 是遥操作器抽象，`config.py` 是选择注册配置，`utils.py` 定义事件和工厂。`so_leader/`、`koch_leader/`、`omx_leader/`、`openarm_leader/`、`openarm_mini/`、`bi_openarm_leader/`、`bi_so_leader/`、`rebot_102_leader/`、`bi_rebot_102_leader/`、`reachy2_teleoperator/` 均为对应实体设备的配置和驱动；`gamepad/`（手柄及工具）、`keyboard/`（键盘）、`phone/`（手机与 `phone_processor.py`）、`homunculus/`（手臂/手套/关节映射）、`unitree_g1/`（外骨骼标定、串口、IK 和 G1 遥操）提供非电机输入设备。

## 8. 训练、分布式、强化学习与奖励

| 目录 | 文件职责 |
|---|---|
| `common/` | `control_utils.py` 控制循环工具；`train_utils.py` 训练循环、checkpoint 和日志；`wandb_utils.py` W&B 集成。 |
| `optim/` | `factory.py` 创建优化器/调度器；`optimizers.py` 优化器实现；`schedulers.py` 学习率调度器。 |
| `distributed/` | `checkpoint.py` 分布式 checkpoint；`factory.py` 初始化并行环境；`parallel_dims.py` 并行维度；`utils.py` 分布式辅助。 |
| `rewards/` | `pretrained.py` 奖励模型基类，`factory.py` 工厂；`classifier/`、`robometer/`、`sarm/`、`topreward/` 各含配置、模型、处理器，后三者另含 `compute_rabc_weights.py`，SARM 另含 `rabc.py`/`sarm_utils.py`。 |
| `rl/` | `actor.py` 策略 actor；`buffer.py` replay buffer；`learner.py`/`learner_service.py` 学习器；`trainer.py`/`train_rl.py` RL 训练；`queue.py` 进程队列；`eval_policy.py` 评估；`gym_manipulator.py` Gym 机器人；`joint_observations_processor.py` 关节观测；`crop_dataset_roi.py` ROI 数据裁剪；`data_sources/data_mixer.py` 数据混合；`algorithms/` 提供算法基类、配置、工厂和 `sac/` 实现。 |

## 9. 推理、部署与通信

`rollout/` 是可编程部署引擎：`configs.py` 配置，`context.py` 运行上下文，`controller.py` 生命周期状态机，`interactive.py` 交互入口，`robot_wrapper.py` 硬件包装，`ring_buffer.py` 实时缓冲。`inference/` 的 `base.py` 是推理引擎接口，`sync.py`/`rtc.py` 是同步和 RTC 实现，`factory.py` 创建引擎；`strategies/` 的 `base.py`/`core.py` 定义策略循环，`episodic.py`、`dagger.py`、`highlight.py`、`sentry.py` 是具体 rollout 策略，`factory.py` 负责选择。

`async_inference/` 实现异步策略服务：`configs.py` 配置，`constants.py` 协议常量，`helpers.py` 公共函数，`policy_server.py` 服务端，`robot_client.py` 机器人客户端。`transport/` 保存 gRPC 定义：`services.proto` 为协议源文件，`services_pb2.py` 和 `services_pb2_grpc.py` 为生成代码，`utils.py` 为通信工具。

## 10. 标注、变换和通用工具

`annotations/steerable_pipeline/` 是可人工干预的 VLM 标注流水线：`config.py` 配置，`executor.py` 执行，`frames.py` 帧抽取，`reader.py`/`writer.py` 数据读写，`staging.py` 暂存，`validator.py` 校验，`vlm_client.py` VLM 调用；`modules/` 下的三个文件分别处理通用 VQA、插话/语音、子任务计划与记忆；`prompts/*.txt` 是对应提示词模板。`data_processing/sarm_annotations/subtask_annotation.py` 负责 SARM 子任务标注。

`transforms/transforms.py` 提供图像/张量增强和组合变换。`utils/` 是轻量横切工具：`constants.py` 字段名和路径常量，`device_utils.py` 设备选择，`import_utils.py` 可选依赖检测，`hub.py` Hub 辅助，`io_utils.py` 文件 IO，`logging_utils.py` 日志，`random_utils.py` 随机种子，`process.py` 进程管理，`decorators.py` 连接状态装饰器，`errors.py` 异常，`collate.py` batch 拼接，`feature_utils.py` 特征处理，`transition.py` 转移结构，`rotation.py` 旋转表示，`bimanual.py` 双臂工具，`robot_utils.py` 机器人辅助，`action_interpolator.py` 动作插值，`sample_weighting.py` 样本权重，`cycle_timer.py` 控制周期计时，`keyboard_input.py`/`stdin_input.py`/`pedal.py` 输入设备，`doctest_utils.py` doctest，`visualization_utils.py` 可视化公共函数，`rerun_visualization.py`/`foxglove_visualization.py` Rerun/Foxglove 输出。

## 11. 作业入口与脚本：`jobs/`、`scripts/`

`jobs/annotate.py`、`jobs/dataset.py`、`jobs/hf.py` 分别编排标注、数据集和 Hugging Face 作业。`scripts/` 中每个 `lerobot_*.py` 都是可执行 CLI：`lerobot_train.py` 训练，`lerobot_eval.py` 评估，`lerobot_record.py` 采集，`lerobot_replay.py` 回放，`lerobot_teleoperate.py` 遥操，`lerobot_rollout.py` 部署；`lerobot_calibrate.py`/`lerobot_setup_motors.py`/`lerobot_setup_can.py` 做硬件初始化；`lerobot_find_cameras.py`/`lerobot_find_port.py`/`lerobot_find_joint_limits.py` 做设备发现；`lerobot_dataset_viz.py`/`lerobot_info.py` 查看数据；`lerobot_edit_dataset.py`、`convert_dataset_v21_to_v30.py`、`lerobot_convert_dcp.py`、`augment_dataset_quantile_stats.py` 做数据迁移/编辑；`lerobot_imgtransform_viz.py` 查看变换；`lerobot_train_tokenizer.py` 训练 tokenizer；`lerobot_annotate.py` 启动标注。

`templates/lerobot_modelcard_template.md` 和 `templates/lerobot_rewardmodel_modelcard_template.md` 是发布模型/奖励模型时使用的 Hub card 模板。

### 文件命名约定与 `__init__.py`

源码中未在上表逐个展开的 `__init__.py` 都是 Python 包初始化文件：它们负责建立包边界、触发可选实现的注册，并在需要时重新导出公共类；通常不包含独立业务流程。具体来说，`annotations/`、`async_inference/`、`cameras/`、`common/`、`data_processing/`、`distributed/`、`envs/`、`jobs/`、`model/`、`motors/`、`optim/`、`rewards/`、`rl/`、`robots/`、`teleoperators/`、`transforms/`、`transport/`、`utils/` 以及每个具体硬件/策略/奖励子目录的 `__init__.py` 都属于这一类；其中 `policies/__init__.py`、`datasets/__init__.py`、`processor/__init__.py`、`robots/__init__.py`、`teleoperators/__init__.py`、`rollout/__init__.py` 还集中导出对外 API。

下列少量容易被忽略的文件也有明确职责：

| 文件 | 作用 |
|---|---|
| `model/__init__.py`、`model/kinematics.py` | 模型工具包入口；运动学/正逆运动学计算。 |
| `data_processing/__init__.py`、`sarm_annotations/__init__.py` | 数据处理包入口；后者导出 SARM 标注实现。 |
| `rewards/*/__init__.py` | 注册对应奖励模型并导出其配置/模型/处理器。 |
| `rl/algorithms/__init__.py`、`rl/algorithms/sac/__init__.py` | RL 算法包和 SAC 实现包入口。 |
| `rollout/inference/__init__.py`、`rollout/strategies/__init__.py` | 推理引擎和 rollout 策略的公共导出。 |
| `policies/groot/action_head/__init__.py`、`policies/fastwam/wan/__init__.py`、`policies/molmoact2/molmoact2_hf_model/__init__.py`、`policies/wall_x/qwen_model/__init__.py` | 各模型内部组件包入口和注册。 |
| `robots/unitree_g1/controllers/__init__.py` | G1 控制器包入口。 |

配置、模型、处理器三件套之外的策略辅助文件也按名称分工：`memory.py` 保存时序记忆，`*_utils.py` 放模型专用工具，`*_interface.py` 封装外部 VLM/LLM 接口，`action_head.py`/`action_hub.py` 实现动作头或动作离散化，`world_model.py` 实现世界模型，`constant.py` 保存模型常量；它们不改变“配置 -> 模型 -> 处理器”的公共接入方式。

## 12. 阅读与修改建议

1. 从 `scripts/lerobot_train.py` 或 `scripts/lerobot_record.py` 进入，查看配置如何组装。
2. 训练问题优先沿 `configs/train.py -> datasets/factory.py -> processor/factory.py -> policies/factory.py -> common/train_utils.py` 阅读。
3. 真机问题沿 `robots/robot.py`、`cameras/camera.py`、`motors/motors_bus.py`、`teleoperators/teleoperator.py` 阅读，再进入具体硬件目录。
4. 部署问题沿 `rollout/controller.py -> rollout/inference -> rollout/strategies -> processor/policy_robot_bridge.py` 阅读。
5. 新增策略/机器人/遥操作器时，通常需要一个配置类、一个实现类、`__init__.py` 注册，并在对应 factory 的延迟导入映射中加入名称。

本文件可用下面的命令检查清单是否仍覆盖当前源码（新增文件后应补充说明）：

```bash
find src/lerobot -type f | sort
```
