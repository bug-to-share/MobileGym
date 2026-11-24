# GUI Agent 任务理解、相关工作与优化思考

研究日期：2026 年 11 月 14 日。用途：为 MobileGym 项目面试准备补充任务理解、方法选择与可检验的研究计划。以下区分“论文方法”“本地已知事实”和“待验证方案”，不把外部论文成绩算作个人成果。

### 9.1 之前调研了什么 这次补充什么

之前已经调研过一轮，来源记录包含 12 篇论文，重点是 progress reward、PRM、信用分配、奖励塑形和在线课程。此前的 改进报告（历史记录） 保留了那一轮结论。

本次扩大到五条主线：任务与评测、视觉 grounding、规划和记忆、数据与后训练、奖励和信用。检索 arXiv、ACL Anthology 等原始论文及官方项目，交叉核对方法、观测/动作接口、评测单位与实际阅读版本。部分近期工作仍是预印本。本报告是有目的的叙述性研究，不声称穷尽全部论文，也不提供跨协议排行榜。

**面试时最值得表达的主张：** GUI Agent 的优化应由失败来源驱动。先建立“感知—决策—执行—验证”的诊断，再决定补数据、改记忆、做 RL 或增加过程信用。你的项目适合进一步发展为“可验证环境下的长链多模态策略学习”。

### 9.2 如何理解 GUI Agent 这个任务

#### 9.2.1 它是带目标的部分可观测决策问题

可以用以下形式理解：

    隐藏状态 s_t：业务数据、页面、导航栈、键盘、焦点、权限、未显示的信息等
    观察 o_t：当前截图；具体系统也可能开放 DOM / accessibility tree
    可见历史 h_t：(o_0, a_0, …, o_t)，或该历史的受控摘要
    目标 g：用户要求及约束
    策略 πθ(a_t | g, h_t)：决定操作、查看、等待、恢复或终止
    环境转移：执行 a_t 后得到 s_(t+1)，再产生新观察

同一截图可能对应不同隐藏状态；环境转移即使确定，也不意味着模型完全知道状态。目标可能要求先读取信息，再在另一 App 中使用，当前画面不足以决定动作。POMDP 视角可从 OSWorld 与 Agent S2 的形式化出发，但上面的具体变量解释是针对本项目的建模。[OSWorld](https://arxiv.org/html/2404.07972v2)、[Agent S2](https://arxiv.org/html/2504.00906v1)

在本地 MobileGym 中，generic_v2 使用当前截图和历史文本。judge 能读取业务状态，是环境的监督能力；它不意味着 policy 也读取了该状态。训练与评测必须分别写清楚：

| 通道 | 能看到什么 | 作用 |
|---|---|---|
| Policy 输入 | 指令、当前图、允许的可见历史 | 产生动作 |
| 执行器 | 动作参数与环境接口 | 将动作落实为点击、输入等 |
| Reward / judge | 任务条件、初态与结果状态 | 评分、成功判断、副作用检查 |
| 诊断 oracle | 人工子目标、正确元素框等额外信息 | 仅用于定位瓶颈，单独报告 |

#### 9.2.2 六种能力共同决定完整任务成功

1. **Grounding：** 已经知道要找哪个元素，能否定位到正确位置？
2. **Planning：** 根据目标和进度，此时应该执行哪个业务操作？
3. **State tracking：** 上一步是否生效，页面是否仍在加载，实际对象是谁？
4. **Memory：** 跨页面和跨 App 后，能否保留正确实体、数值与约束？
5. **Recovery：** 操作失败或进入错误页面后，能否识别并恢复？
6. **Termination：** 什么时候应继续、核验、完成或中止？

例如“把聊天中确认的会议地址加入日历”，可能先涉及信息读取，再涉及跨 App 记忆、字段输入和保存验证。点击位置全部正确，也可能因为日期记错而失败。这是示意任务，不是新增实验。

可以用一个简化算式解释长链困难：若每个关键决策的条件成功率都为 95%，20 个关键决策全部成功约为 0.95²⁰＝35.85%。这是恒定条件成功率、无恢复的教学模型，不是项目成功率预测；真实错误存在依赖，恢复操作也会改变结果。

#### 9.2.3 优化目标应同时考虑结果、成本和约束

当前项目的严格成功要求目标达成、没有不允许的副作用并正确 COMPLETE。产品层面还需考虑交互次数、推理 token、延迟和可恢复性。一个实用研究目标是：

    在固定交互与计算预算、相同观测/动作接口下，提高严格任务成功率；
    同时报告成本、终止分布、环境错误和非预期状态修改。

不要直接用“最短轨迹”替代成功：检查信息、等待加载和纠错可能需要合理的额外操作。也不要一开始把许多成本项混成单个奖励；先测清它们与成功的关系，再用消融决定权重。

### 9.3 代表性工作教会我们什么

下面按问题组织文献，而不是把不同基准的最好分数放在一起。版本与阅读范围见文末来源索引。

#### 9.3.1 任务和评测：测到的是哪一种能力

| 工作 | 原始研究的关注点 | 阅读时必须保留的区别 |
|---|---|---|
| [AndroidWorld](https://arxiv.org/html/2405.14573v5) | 参数化移动任务、初始化与设备状态判分 | Android emulator 中的真实应用；环境可提供截图和 a11y，不保证所有基线纯视觉 |
| [OSWorld](https://arxiv.org/html/2404.07972v2) | 桌面真实应用中的执行和功能正确性 | 主 369 任务是 Ubuntu；观测、PyAutoGUI 动作与预算需固定 |
| [WebArena](https://arxiv.org/html/2307.13854v4) | 自托管网站、业务任务与功能评测 | 模板与任务实例不同；可开放 DOM / 元素 ID，部分评测含模型匹配 |
| [VisualWebArena](https://arxiv.org/html/2401.13649v2) | 需要视觉信息的在线网页任务 | SoM 元素框与 ID 来自 JavaScript；图像输入不等于原始截图加坐标协议 |
| [AndroidControl](https://arxiv.org/html/2406.03679v6) | 演示数据规模、不同泛化划分 | 原论文基线使用 a11y 文本，主要是离线逐步动作评测 |
| [ScreenSpot-Pro](https://arxiv.org/html/2504.07981v1) | 高分辨率界面上的目标定位 | 点落在目标框内，与在线任务完成是不同指标 |
| [OSWorld 2.0](https://arxiv.org/html/2606.29537v2) | 长工作流、动态环境、结果检查点 | 终态检查点部分分不自动构成时间逐步 PRM |

**对本项目的推论：** 建立三层评测：固定截图定位 → 固定可见前缀的一步动作 → 在线完整任务严格成功。前两层帮助定位原因，第三层检验策略在自身产生的状态上能否完成目标。不能把离线准确率当作在线能力的替代品。

#### 9.3.2 视觉与规划：专门定位器和端到端策略各有用处

| 工作 | 核心思路 | 对 MobileGym 的可借鉴部分 |
|---|---|---|
| [SeeClick](https://arxiv.org/html/2401.10935v2) | 用 GUI 元素与坐标监督学习视觉定位 | 利用模拟器产生训练标签；评测策略仍只读截图 |
| [UGround](https://arxiv.org/html/2410.05243v3) | 规划器描述目标，专门 grounding 模型定位 | 固定正确目标描述，测独立定位器能修复多少失败 |
| [ShowUI](https://arxiv.org/html/2411.17465v1) | 视觉 token 选择及图像/动作交错建模 | 先试近期图像与关键帧，再考虑改模型内部压缩 |
| [Agent S2](https://arxiv.org/html/2504.00906v1) | 专家 grounding 与主动分层重规划 | 将计划、执行和验证拆开诊断；计入额外模型调用 |
| [UI-TARS](https://arxiv.org/html/2501.12326v1) | 原生视觉策略、反思修正数据及 Agent DPO | 收集真实错误前缀下的纠正动作和恢复示范 |

**选择方法的依据：** 如果给定正确目标描述仍点错，优先处理定位或坐标映射。如果同一状态下给定子目标能明显改善动作，规划或目标表达可能是瓶颈。这样的 oracle 提示是诊断工具，其结果不能并入正常任务成功率；还需控制随机性、提示长度和描述歧义。

#### 9.3.3 数据和后训练：算法名称不足以说明训练方式

| 工作 | 核验到的训练/数据机制 | 不能据此直接推出什么 |
|---|---|---|
| [OS-Genesis](https://aclanthology.org/2025.acl-long.277/) | 先探索状态转移，再反向合成任务，筛选轨迹 | 生成的语言任务天然可执行、无标注成本 |
| [UI-TARS-2](https://arxiv.org/html/2509.02544v2) | 迭代数据流程与多轮 PPO；value 预训练、GAE、轨迹池 | 它使用 GRPO；其所有结果都在纯 GUI 接口取得 |
| [InfiGUI-R1](https://arxiv.org/html/2504.14239v1) | Actor2Reasoner、推理蒸馏/SFT，随后 RLOO | 子目标文本评分等于真实业务状态完成 |
| [GUI-R1](https://arxiv.org/html/2504.10458v4) | 对固定观测的动作输出做规则奖励 GRPO，筛选训练问题 | 3K 样本就是全部数据成本；离线 step SR 等于在线任务 SR |
| [OpenMobile-2](https://arxiv.org/html/2610.14157v1) | 扩展环境和演示数据；GUI / hybrid SFT 与可验证 RL | GUI + app-native tools 成绩与纯截图坐标策略可直接比较 |

这些研究提供方法选择，不能组成“同一个基准上的先后胜负”。例如 UI-TARS 原版的在线轨迹收集与偏好训练、UI-TARS-2 的 PPO，以及 InfiGUI-R1 的 RLOO，应分别解释。

### 9.4 有哪些手段 如何决定先做什么

#### 9.4.1 观测、记忆与执行：先验证低成本改动

当前历史文本可能没有记录未来要用的数值。建议先冻结模型，做以下对照；它们都是新增方案：

| 方案 | 预期作用 | 验证与代价 |
|---|---|---|
| 当前图 + 原文本历史 | 保留现有基线 | 保存实际模型输入 |
| 当前图 + 最近 2 张图 | 识别转移、近一步误操作 | 图像 token、时延、相同任务重复评测 |
| 当前图 + 关键历史图 | 跨 App 保留源信息 | 关键帧选择错误、旧状态误用 |
| 近期图 + 结构化摘要 | 控制长上下文成本 | 摘要事实错误、丢约束、递归误差 |

摘要可以包含“目标实体、已确认值、证据 step、已完成事项、待做事项”，并区分观察与推测。只允许从模型可见记录生成，不能把 judge 的隐藏字段直接写进部署记忆。ShowUI 与 UI-TARS-2 提供了历史建模的不同实现，但上述四组是针对本项目提出的对照设计。[ShowUI](https://arxiv.org/html/2411.17465v1)、[UI-TARS-2](https://arxiv.org/html/2509.02544v2)

执行层同样需要检查：坐标是否与该帧尺寸匹配，键盘是否遮挡，点击后是否等待稳定画面，超时重试是否重复提交。截图相同不一定代表动作未生效，截图变化也不等于业务完成。先把执行故障单列，避免用更多 RL 更新去拟合执行器问题。

主动查看、滚动找元素、打开详情、等待后重看可以具有信息价值，但都消耗预算。最初应测“额外查看减少了哪一种不确定性”，不要给每次查看固定正奖励。

#### 9.4.2 数据：增加技能覆盖，而不仅是改写指令

数据可沿五个维度扩展：任务目标、参数实体、界面布局、状态起点、失败与恢复路径。训练任务 ID 或文本变多，并不自动意味着技能覆盖变多。

适合本项目的做法：

- 利用声明式导航和可控状态生成候选任务，执行并通过 judge 后才收录。
- 生成目标相近但约束不同的任务，要求核对实体、日期或副作用。
- 记录真实失败状态后的恢复示范，不能假定一次 BACK 总能撤销业务操作。
- 将跨 App 数据依赖作为单独维度：源信息如何获得、何时使用、是否过期。
- 合成与审核只使用训练/开发任务，最终 holdout 不参与造数据或改规则。

OS-Genesis 的“先探索再构造任务”提供一种数据入口；它使用模型进行任务生成和轨迹质量判断，因此仍需要把教师成本、失败候选和验证开销纳入统计。[OS-Genesis](https://aclanthology.org/2025.acl-long.277/)

#### 9.4.3 SFT、偏好优化和 RL 的分工

| 方法 | 必需数据条件 | 擅长提供的训练信号 | 本项目的入口 |
|---|---|---|---|
| SFT | 可靠、可执行的正例步骤 | 格式、定位、基本技能和恢复行为 | 训练集内成功/恢复示范 |
| DPO | 同可见上下文下的可信正负响应 | 对错误与纠正动作的相对偏好 | 相同前缀的错误动作与复核修正 |
| 固定观测的动作 RL | 动作类型、目标框等标签 | 一步动作输出的规则奖励 | 独立诊断或 warm-start |
| 在线 GRPO | 同任务多条交互轨迹和终态奖励 | 完整任务的相对表现 | 当前已运行路径 |
| PPO + value | 在线轨迹及价值估计训练 | 基于 return/TD 的时序优势 | 需新增 value 训练、校准和稳定性验证 |

恢复 SFT 中可以把错误步骤保留为上下文，只对已核验的纠正步骤施加正监督。DPO 不能随意配对两个不同截图下的动作。在线 RL 有机会遇到策略自身诱发的状态；全失败且同奖的组仍缺乏组内任务奖励信号，因此可能需要示范、课程或额外过程信息。

UI-TARS 的纠正数据/偏好流程、GUI-R1 的固定观测动作奖励，以及 UI-TARS-2 的多轮 PPO，分别说明了不同数据条件，不能用“都做了 RL”抹平差异。[UI-TARS](https://arxiv.org/html/2501.12326v1)、[GUI-R1](https://arxiv.org/html/2504.10458v4)、[UI-TARS-2](https://arxiv.org/html/2509.02544v2)

### 9.5 Reward、PRM 与信用分配 应该怎么区分

#### 9.5.1 先回答你当前的 reward 是什么

本项目当前是**规则终态奖励**：根据最终目标检查、终止类型和副作用等计算轨迹分数。progress 能给部分分，但这仍然是终态结果分解。严格 result_only 与错误 COMPLETE 清零是已有本地分支。

默认 verl 路径读取 trajectory reward，计算组内优势，再广播到同条轨迹的响应步骤。Step.reward 的存在不代表该值已经参与默认优势计算。[本地优势路径](../mobilegym-rl/rllm/experimental/verl/verl_advantage.py)

| 概念 | 输出/作用 | 是否依赖学习得到的评分器 |
|---|---|---|
| 规则 outcome reward | 最终目标是否满足，或终态分项进度 | 否 |
| 规则过程奖励 | 动作前后某个可验证子目标变化 | 否 |
| ORM | 学习预测整条轨迹结果 | 是 |
| PRM | 对步骤、前缀或转移提供学习型过程评分；标签语义由方法定义 | 是 |
| Credit assignment | 将结果/过程信息转为不同决策的学习信号 | 不一定；可基于规则、value、分组或重分配 |
| 推理时 verifier | 对候选动作或轨迹筛选/排序 | 可为规则或模型；不必更新 policy |

每一步都有数字，不自动等于 PRM；有 PRM 也不自动等于训练时逐步信用分配。要继续追问：它预测正确性、成功概率还是进展？用于重排还是 RL？最终如何进入 advantage？

#### 9.5.2 代表论文之间的关键区别

- **ProgRM：** 学习 GUI 进展评分，说明进度标签是可研究的监督对象。其接口与评测设置需单独核对，不能直接套用 MobileGym 的业务状态语义。[ProgRM](https://arxiv.org/html/2505.18121v1)
- **GuidNav：** 过程评分用于推理时候选选择；这条路线增加选择预算，不等于已经训练出更好的 policy。[GuidNav](https://arxiv.org/html/2504.16073v1)
- **GUI-Shepherd：** 将过程评价与长序列验证结合，并研究在线 PPO/GAE；需要核对奖励来源和参考轨迹条件。[GUI-Shepherd](https://arxiv.org/html/2509.23738v1)
- **GiGPO：** 在 episode 组之外寻找可比较的 anchor state 组，增加局部相对优势。v3 还包含视觉小环境附录实验，但不是移动 GUI 证据；截图相似也不足以证明状态、历史和剩余预算相同。[GiGPO](https://arxiv.org/html/2505.10978v3)
- **AgentPRM：** 区分前缀的 promise 与 progress。本次核对发现其附录 RL 实验使用最后一步 PRM 得分作为整轨迹奖励；不能据此声称它直接验证了逐步 dense policy update。[AgentPRM](https://arxiv.org/html/2511.08325v1)

这些是对原文使用方式的辨析，论文自身收益不能替代本项目对照。新增状态规则过程信号应先叫“可验证过程奖励”；以后训练只看可见图像/历史的评分器，才进入学习型 PRM 的问题。

#### 9.5.3 为什么把 progress 做差分还不够

设 p_t 为第 t 步后的子目标进度。若定义：

    r_t = p_t − p_(t−1)
    Σ_t r_t = p_T − p_0

求和会消去中间项。如果仍把这个总和变成一个轨迹优势并广播，步骤间依旧没有由过程进度产生的差异化优势；共享同一个优势值，并不意味着不同步骤的策略梯度相同。要完全复现旧奖励，还须保留初始进度以及原先的终止、答案和副作用处理；不能把上式直接当现有 reward 的等价改写。

势函数塑形的标准形式是 F_t＝γΦ(s_(t+1))−Φ(s_t)。折扣求和后为 −Φ(s_0)＋γ^TΦ(s_T)。在相同起点、终止势为零等条件下，它只是一个常数；继续用于组内终态广播时可能被中心化抵消。一般的正向进度奖金并不自动拥有策略不变性保证。[Ng、Harada 与 Russell，1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)

#### 9.5.4 四条真正需要比较的信用路线

| 路线 | 如何区分步骤贡献 | 代价与风险 |
|---|---|---|
| 子目标段信用 | 将可验证子目标变化归入对应执行片段 | 切段质量；必要导航零即时进度；改变辅助目标 |
| Value / GAE | 根据可见历史估计未来回报，形成时序差分优势 | 价值误差、额外训练与显存、稀疏回报 |
| Anchor-state 局部分组 | 比较足够相近决策条件下的不同后续结果 | 重复状态覆盖低、隐藏状态混淆、条件不等价 |
| 同前缀分叉 rollout | 在可恢复状态上尝试不同动作并估计后续成功 | 克隆真实性、Monte Carlo 成本、后续策略偏差 |

本项目优先研究第一条。一个可消融的起点是：

    A_(episode,t) = A_outcome(episode) + λ · C_segment(episode,t)

C_segment 要有明确的事件定义、范围和归一化规则；λ＝0 必须恢复原数值与 mask。该式是待验证的混合辅助目标，不宣称无偏或保证最优策略不变。

落地要贯通“执行前后记录 → episode/step/trace 唯一映射 → advantage → response mask”。循环检测有可能在 env.step 之前退出，最后生成动作未实际执行；这类事件必须单列。第一轮先关闭只看终态奖励差异的动态筛选，否则含过程差异的同终态组可能被丢掉。

中间态分叉应先验证 UI、业务状态、导航栈、焦点、键盘和时钟恢复。MobileGym 论文描述的 snapshot/fork 能力，不等于当前训练入口已经验证任意中途状态的精确复现。[MobileGym](https://arxiv.org/html/2605.26114v1)

### 9.6 结合你的实验 应该优先优化什么

本节是下一轮方案，尚未运行。历史最好点跨 App 为 6/93，单 App 为 50/163；训练跨 App 占比 19/160，验证占比 93/256。它们提示应优先检查任务覆盖和信息保持，但不能单凭相关性认定“缺记忆”就是根因。FCzero 后期 ABORT 增多、成功率回落，也不能只用奖励曲线解释。

#### 9.6.1 从失败记录到具体干预

| 可观察的失败 | 先检查什么 | 可单独尝试的改进 | 验证指标与反例 |
|---|---|---|---|
| 点错按钮、点在空白处 | 坐标缩放、裁剪、目标可见性、动作是否执行 | grounding SFT；局部裁剪；独立 grounder | 同一意图下命中率；小目标/高分辨率切片；裁剪漏上下文 |
| 点对了错误功能 | 指令是否理解、实体是否消歧 | 子目标表示；动作前目标核对 | 计划正确率、错误对象率；不能用命中框率替代 |
| 跨 App 后忘记姓名、金额或内容 | 输入历史是否保留来源证据 | 最近截图；结构化任务记忆 | 跨 App 严格成功；记忆事实错误；token/延迟 |
| 长时间滚动、重复跳转 | 页面变化检测、历史和动作结果是否可见 | 变化摘要、重复状态提示、恢复数据 | 重复动作率、恢复成功、ABORT；少循环不必然多成功 |
| 中途走错后无法恢复 | 训练数据是否只有理想轨迹 | 从真实错误状态构建恢复 SFT | 固定扰动后的完成率；正常任务是否受损 |
| 提前 COMPLETE、误报成功 | 模型能否看到完成证据，终止语义是否一致 | 可见证据核对、终止校准数据 | false-COMPLETE 与漏报；严格成功优先 |
| 动态采样经常补满仍无有效组 | 全成功/全失败组比例、任务难度、预算错误 | 难度分层课程；调整组大小的独立对照 | 有效组/候选数、困难任务覆盖、总环境动作 |
| 正确前缀和致错动作得到相同权重 | 是否真正消费逐步 advantage | 子目标段信用；可靠 anchor 分组；value/GAE | 非零局部优势覆盖、正确/错误动作权重、hold-out 成功 |
| 长任务几乎全失败 | 任务可执行性、预算、数据覆盖、能力组合 | 合法子任务课程；逐步增加跨度 | 同预算下长任务成功；不能只增加步数再宣称算法增益 |
| 后期掉点且环境 ERROR 增多 | 环境重置、并发、超时、服务异常 | 先修复可重复的执行问题 | 原始总体成功 + ERROR 分布；禁止删失败再报最好结果 |

这里的自动失败分类应抽样人工复核。一次失败可能同时含定位、状态判断和计划错误；用“最早可确认偏离点”标主因，允许多标签，避免把末端超时当根因。

**低成本诊断可以先做三个 oracle 对照。** 仅在诊断集提供正确下一步意图、正确目标坐标或正确历史事实，分别观察失败是否缓解。它们改变了观察/监督条件，只用于定位瓶颈，不得计入正式纯视觉成绩；其中“正确意图”也可能包含额外信息，因此不是干净的能力因果分解。

#### 9.6.2 建议执行顺序与停止条件

| 阶段 | 设计 | 支持继续投入的信号 | 停止或转向条件 |
|---|---|---|---|
| P0：恢复可信基线 | 固定环境版本、FCzero 规则、任务参数、预算、初态和评测流程；重跑冻结 checkpoint | 结果可复现，ERROR 与任务难度可分辨 | 重置/执行不稳定时先修工程 |
| P1：冻结模型的记忆对照 | 当前文本历史 vs 最近 2–3 张截图 vs 结构化记忆 vs 两者组合 | 跨 App/信息搬运改善，正常任务无明显损失，成本可接受 | 摘要幻觉或视觉 token 开销抵消收益 |
| P2：恢复数据 SFT | 相同基座、相同总训练 token；普通示范与加入恢复轨迹对照 | 固定错误状态能恢复，完整任务成功也改善 | 只改善人工扰动，正常起点无收益；错误前缀被模仿 |
| P3：局部信用可行性 | 保持 outcome 基线；先选一种规则片段或 anchor 辅助项 | 梯度映射正确、覆盖充分、同预算严格成功改善 | anchor 低覆盖或合组不可靠；子目标可被刷分 |
| P4：学习型 PRM | 已验证监督语义后，训练图像+历史评分器；先离线与冻结 actor 重排 | OOD 校准、候选区分及端到端收益均成立 | 离线排序高分却诱发奖励利用，或 scorer 成本过高 |

P1 是推理接口干预，P2/P3 是训练干预，分别报告。近期多图的收益也可能来自额外计算，因此另报告相同 token 上限的压缩策略，而不强行声称所有输入完全等价。

第一轮选 16 个训练侧任务、每题 8 条轨迹，共 128 条，可以检查标签、覆盖和映射；这是工程诊断预算，没有足够统计功效的保证。先证明局部项系数为零时与基线一致，再做在线实验。每次只增加一个主要因素，保留方案再用至少 3 个独立训练 seed 和冻结测试集复核；3 个 seed 也不是充分精度的保证。

**我会先做 P0→P1，并根据失败类型选择 P2 或 P3。** 现阶段还没有“PRM 是性能瓶颈”的证据；项目已有可查询状态，先用它验证过程标签是否真的有用，能降低错误方向的投入。

#### 9.6.3 如果要做课程与样本效率优化

WebRL 提供失败驱动的课程与回放思路；MobileRL 把成功轨迹长度、困难正例与难度过滤结合。它们的输入接口和优化器都与当前路径不同，适合拆成单因素实验。[WebRL](https://arxiv.org/html/2411.02337v3)、[MobileRL](https://arxiv.org/html/2509.18119v2)

建议按任务家族、跨 App 与难度分层采样，保留最难层的最低探索配额。仅对成功轨迹奖励效率，防止提前 ABORT 因长度短而获益。若引入历史正例回放，应保存行为策略信息并明确 off-policy 处理；不能把不同策略的旧轨迹直接当成当前同组 rollout。补采样减少了无效更新，也可能增加大量环境开销，必须同时看动作、token、GPU 时间和总墙钟时间。

### 9.7 怎样证明优化有效 评测应如何设计

#### 9.7.1 至少区分四层泛化

| 层次 | 划分单位 | 回答的问题 |
|---|---|---|
| 参数泛化 | 同任务族的新实体、金额、日期和随机初态 | 是否记住具体答案或对象？ |
| 模板/任务族泛化 | 未参与训练和选点的任务语义组合 | 能否组合已有技能？ |
| App/界面泛化 | 未见 App、布局变化、主题、语言、分辨率 | 视觉和操作规则能否迁移？ |
| 平台/真实环境迁移 | 浏览器模拟器到 Android/桌面等 | 是否依赖模拟器时延、布局或状态规则？ |

这四种难度不构成普遍严格排序。task ID 不相交只证明 ID 层面的分离；模板、业务流程、画面或生成器仍可能重合。当前反复用于选点的验证集应当视为开发集，另冻结一个最终测试集。所有数据生成器、PRM 标签和课程筛选都不能读取其答案或结果来调参。

#### 9.7.2 统一报告表

| 维度 | 应报告的内容 |
|---|---|
| 主结果 | 固定预算下 strict pass@1，成功数/总数；按任务族、单/跨 App、难度分层 |
| 部分进度 | 与严格成功并列，说明终态 check 的语义，不能换名为成功率 |
| 稳定性 | 多训练 seed 的均值/离散度；同 checkpoint 多环境初态的重复测量 |
| 错误与终止 | ERROR、超时、循环、ABORT、false-COMPLETE；统一保留总体分母 |
| 效率 | 环境动作、生成/输入 token、墙钟时间、GPU 小时、PRM/teacher 调用；包含丢弃 rollout |
| 恢复 | 固定扰动下成功率、额外动作；同时保留正常起点主结果 |
| 搜索 | 候选数、重试数、verifier 选择规则；与单次策略执行分开报告 |
| 选择过程 | checkpoint 选择规则、验证次数、停止规则；避免只挑最好点 |
| 能力保持 | 固定 grounding/动作集、简单任务和长任务，检查后训练的能力退化 |

相同动作预算不等于相同计算预算，候选重排与多图历史尤其如此。可以同时给“同动作预算”和“同 token/GPU 成本”两条性能曲线，寻找成本—成功率的折中。成功轨迹平均长度有幸存者偏差，应同时报告全体 episode 的动作数和失败终止分布。

配对任务 bootstrap 可以估计任务采样的不确定性；有模板相关性时按任务家族做 cluster bootstrap 更合理。它不能替代训练 seed 的方差，也不能把一次训练的多个 checkpoint 当独立训练样本。pass@k 要明确重复采样和候选选择条件；单次成功率不能在不满足条件时直接换算为真实多次尝试能力。

#### 9.7.3 Judge 与环境也是实验对象

基于状态的规则 judge 可检查真实业务结果，但仍可能存在遗漏。应设计少量针对性的反例：错误对象、只改页面不改数据、完成后撤销、重复执行、部分字段正确、提前 COMPLETE、跨 App 副作用。对照人工轨迹审计，检查假阳性与假阴性。

完整重置应验证业务状态、导航栈、焦点、键盘、模拟时间和权限等。评测时隐藏状态可由 judge 使用；若把隐藏状态直接传给 actor，必须另标输入协议。只统计“环境正常”的成功率会掩盖系统可靠性：可补充条件分析，但主表仍保留所有预先指定任务和失败原因。

### 9.8 值得延伸思考的方向

#### 9.8.1 端到端模型还是模块化系统

SeeClick/UGround 把定位作为可训练接口，Agent S2 强调分层规划与专家模块；UI-TARS 系列则探索更统一的视觉动作模型。模块化方便定位错误、替换 grounder 和控制成本，但模块之间的目标描述可能丢信息；端到端减少手工接口，也更难区分能力瓶颈。[UGround](https://arxiv.org/html/2410.05243v3)、[Agent S2](https://arxiv.org/html/2504.00906v1)、[UI-TARS](https://arxiv.org/html/2501.12326v1)

对本项目可先保持同一个 4B actor，增加可审计的任务记忆与动作结果记录；若确认大多数错误来自定位，再考虑专门 grounder。比较时固定高层计划，并把额外模型、调用次数与延迟计入成本。

#### 9.8.2 有 API 还需要 GUI Agent 吗

GUI 为没有开放 API 或业务接口不统一的应用提供可见操作入口；API 则能减少部分操作的歧义和步数。二者可以组成混合系统，但动作空间与可靠性条件变了。

OpenMobile2 将模拟移动环境、模拟器环境和 App 原生工具结合，适合作为“纯视觉基准之外的产品路线”阅读。其工具调用成绩不能直接作为纯截图坐标策略的同协议比较；MobileGym 中供任务初始化/判题的内部 API 也不是 actor 自动获准使用的工具。[OpenMobile2](https://arxiv.org/html/2610.14157v1)

可以设计独立实验：纯 GUI、GUI+同范围公开工具、工具失败后的 GUI fallback。比较严格完成、时延、工具权限、失败可恢复性，而不是只用动作数更少证明模型能力更强。

#### 9.8.3 既然有模拟器，还要不要学 world model

WM-R1 用学习的 GUI 世界模型生成可交互的后续状态，并支持想象与策略学习。这能扩充训练交互，但策略收益受转移预测、渲染一致性和结果评价误差影响；生成画面像真的，不代表业务状态真的正确。[WM-R1](https://arxiv.org/html/2608.27508v1)

MobileGym 已有程序实现的环境和结构化状态，当前先利用可验证的真实环境 rollout 通常更直接。若长尾 App 或真实设备交互昂贵，再考虑 learned world model，并做“一步预测一致性→多步漂移→真实环境策略收益”的逐级检验。世界模型中的成功率不能替代真实或程序环境成功率；训练策略还可能主动利用模型漏洞。

#### 9.8.4 Sim-to-real 的关键是覆盖什么差异

需要分别检查布局/字体、设备尺寸、软键盘、异步等待、系统弹窗、权限、网络错误、应用状态和动作时延。对这些因素做可控随机化，有助于研究稳健性；随机化范围应来自目标设备的实际差异，过度随机化也会增加学习难度。

MobileGym 论文对其自身平台的迁移证据不等于你的 checkpoint 已完成真实设备验证。简历应保留“浏览器模拟移动环境”口径；有相同任务语义、明确映射和真实设备复测后，再写 sim-to-real 结果。[MobileGym](https://arxiv.org/html/2605.26114v1)

#### 9.8.5 不确定性、停止和交互安全

停止也是动作选择的一部分。模型应能区分“完成证据充分”“仍可恢复”“无法完成”；仅增加 ABORT 惩罚可能诱发胡乱点击，仅惩罚 false-COMPLETE 也可能增加放弃。应同时观察严格成功、误报、漏报、恢复和成本。

在真实应用中，页面内容还可能包含与用户目标冲突的指令；可研究将页面视为数据、限制高影响操作、在不确定时请求用户确认。但这应另设任务与威胁模型，本项目尚未验证相关能力，不能写成已经实现的安全保障。

### 9.9 面试延伸问答 12 个能展开讨论的问题

以下 R01–R12 是在前面 30 个项目问答之外新增的研究问题。回答用“已经观察到什么—如何解释—如何证伪”组织；涉及改进时明确是计划。

#### R01 你怎样定义 GUI Agent，它和普通视觉问答有什么不同？

**回答思路：** 它是通过有限观察持续改变环境状态的决策任务。截图只显示状态的一部分，动作会影响后续可见信息、可执行选项和最终业务结果，所以要联合考虑 grounding、状态估计、计划、恢复和终止。视觉问答主要评价当前答案，GUI 还要评价整段交互的结果与代价。

**追问准备：** 用跨 App 复制联系人信息举例：源页面离开后关键信息不可见，记忆直接影响后续动作。可对照当前截图、近期多图和结构化记忆；不能只因为跨 App 分数低就断言记忆不足。

#### R02 Grounding 准确率很高，为什么整任务成功率仍然很低？

**回答思路：** “点中了所说的按钮”不保证“这个按钮是当前该点的”。还存在目标理解、动作顺序、隐藏状态、执行时序和终止错误。长序列会积累失败风险，但独立同分布误差模型只是解释用的简化。

**追问准备：** 将动作意图与坐标分开标注，使用正确意图/坐标的诊断对照，找最早偏离点。不要把离线 point-in-box 指标和在线 strict pass@1 放在一个榜单排序。

#### R03 给模型更多历史截图一定更好吗？

**回答思路：** 历史可能补充页面变化和已离开的实体，也会带来视觉 token、干扰和上下文压缩问题。关键是保留与当前决策相关的证据，而不只是增加窗口长度。

**追问准备：** 冻结 actor 比较文本历史、最近截图、结构化记忆、混合记忆；测跨 App 成功、事实保持与成本。记忆摘要应标注来源步骤和置信度，错误信息需要可更新；不能把 ground-truth 隐藏状态写进纯视觉记忆。

#### R04 SFT、DPO 和在线 RL 分别适合解决什么？

**回答思路：** SFT 建立接口、基础技能和恢复行为；DPO 需要同一上下文下可解释的偏好对；在线 RL 在可执行环境中直接优化结果和交互决策。三者的数据来源与失败模式不同，不能用算法名称替代问题诊断。

**追问准备：** 如果格式错误很多，先做数据与 SFT；基础行为可用但完整任务欠佳，再考虑在线反馈。UI-TARS 的 Agent DPO、InfiGUI-R1 的 RLOO 与 UI-TARS-2 的 PPO 应准确区分，不能统称 GRPO。

#### R05 你现在的 reward 是 PRM 吗？有部分分为何还不算？

**回答思路：** 当前是最终状态的规则评分，可以给部分目标达成分，但没有使用学习得到的过程评分器。PRM 需要说明训练数据、预测对象与使用位置，也可以复用已有评分模型；逐步调用一个终态检查也只是规则过程信号。

**追问准备：** 继续检查 reward 怎样进入 advantage。当前路径将轨迹优势广播；即使新增 Step.reward 字段，也要贯通训练消费链路，才能声称实现逐步信用。

#### R06 为什么把 progress 相邻作差，不一定改进信用分配？

**回答思路：** 差分求和仍是终点减起点，如果最后又变成一个轨迹标量并广播，没有新增由过程进度产生的动作优势差异；各动作本身的梯度仍可能不同。信用分配要使不同决策获得不同的回报估计或 baseline。

**追问准备：** 画出 reward→return→advantage→token mask 的链路，给出 telescoping 等式。势函数塑形的策略不变性有条件；新局部辅助项应做零系数一致性与副作用审计。

#### R07 同一截图下比较动作，是否就是 GiGPO？

**回答思路：** 不够。GiGPO 的局部比较依赖足够可比的决策状态与后续 return。相同截图可能对应不同表单草稿、任务参数、历史信息或剩余预算；视觉相似可能制造错误的优势。

**追问准备：** 先统计可靠 anchor 的覆盖、不同轨迹数和 return 方差。若严格匹配后覆盖很低，就转向子目标段或 value；不能通过随意粗化状态来获得漂亮覆盖率。其视觉小环境实验也不能直接证明手机 GUI 迁移。

#### R08 为什么不直接训练一个更大的 PRM？

**回答思路：** 更大的模型不会自动消除标签含糊、隐藏信息缺失和奖励利用。当前环境已经能提供可验证状态，先证明过程标签优于 outcome 基线，再判断是否需要把它蒸馏为图像评分器。

**追问准备：** PRM 离线相关性、校准、候选排序和在线策略收益分别测。冻结 actor 的重排收益包含额外搜索预算；AgentPRM 的末步轨迹奖励实验也不能当作 dense 更新证据。

#### R09 当前实验支持哪些结论，不支持哪些？

**回答思路：** 同一 FCzero run 在固定任务 ID 与题目文本下，从 23/256 到峰值 56/256，支持该次训练过程中验证表现改善。跨 App 与高难任务仍弱，后期回落提示稳定性和终止行为值得追踪。

**追问准备：** 不能由此证明 FCzero 优于普通 GRPO，不能报多 seed 稳定提升，也不能把六组配置差异较大的历史 run 当严格消融。补做共同初始化、同协议、等预算对照及新 hold-out。

#### R10 多采几个候选再让 verifier 选，算性能优化吗？

**回答思路：** 算系统推理优化，但要分别报告额外计算收益和 policy 参数更新收益。候选、重试和环境回退次数都应纳入预算，且 selector 只能使用协议允许的信息。

**追问准备：** 比较单次 actor、同候选数随机选择、verifier 选择，使用相同初态和预算；没有可恢复状态时，不能假装各候选都在同一状态真实执行过。

#### R11 如果面试官说模拟器太简单，你怎么回答？

**回答思路：** 模拟器的价值是可控初态、状态判题、可复现交互和低成本实验，适合研究后训练机制。它与真实设备在异步、布局、应用业务和错误分布上可能不同，需要单独验证外部有效性。

**追问准备：** 先在模拟器内做受控扰动与 held-out App，再选有明确任务对应关系的真实设备测试。当前应坦诚尚未得到本 checkpoint 的真实设备迁移结果。

#### R12 资源只够再做两组实验，你会怎么选？

**回答思路：** 先修到冻结基线可复现，这是测量前提。然后根据失败审计选一个主因素：若信息丢失占主因，做原历史 vs 近期多图/记忆；若过程监督明确且映射可靠，做原 GRPO vs 单一局部信用项。

**追问准备：** 固定 task、初态、初始化、奖励和预算，预先指定 strict pass@1 与跨 App 分层结果，报告成本和失败模式。两组、单 seed 的结果只用于方向筛选，不承诺统计显著或最终产品收益。

### 9.10 阅读路线、来源与检索边界

#### 9.10.1 建议按问题阅读，而不是背排行榜

1. **任务是什么：** AndroidWorld + MobileGym。先看 observation/action、初始化、judge 与预算，画出训练闭环。
2. **模型究竟哪里出错：** SeeClick + ScreenSpot-Pro + UGround。理解 grounding 的独立指标与高分辨率条件。
3. **数据如何成为能力：** UI-TARS + OSGenesis。区分生成指令、获得成功轨迹、恢复数据与偏好对的成本。
4. **在线训练怎样改变行为：** DigiRL + MobileRL + UI-TARS-2。比较 value/AWR、轨迹 GRPO 与 PPO/GAE；核对输入协议。
5. **怎样给动作分配信用：** GiGPO + GUI-Shepherd + AgentPRM。追踪标签、奖励、advantage 和推理使用方式。
6. **研究边界如何扩展：** Agent S2 + OSWorld2 + OpenMobile2 + WM-R1。讨论长任务、模块化、混合工具与世界模型。

每篇建议做一张卡：任务/接口、监督来源、算法实际更新、主指标、额外成本、最关键限制、一个可迁移实验。面试重点是说明为什么选择某种机制，以及什么证据会让你放弃它。

<a id="scope"></a>

#### 9.10.2 检索方式与证据强度

这是截至 2026-11-14 的**定向叙述式调研**，以原有 12 篇奖励/信用资料为种子，补检 GUI grounding、长任务、数据合成、SFT/DPO、在线 RL、记忆、评测和环境。通过搜索与直接访问核对 arXiv、ACL Anthology、会议页面及作者材料；综述只用于发现线索，方法结论以原始论文为依据。

阅读范围是摘要、方法、实验设置、相关表格和必要附录，逐篇记录版本；没有声称逐字读完所有论文、复现作者代码或检索穷尽。不同基座、数据、输入、动作接口、重试预算的成绩不做统一排名。2026 年近期预印本也不自动代表已经独立复现或达到稳定共识。

访问限制包括 MobileRL 正式会议 PDF 的浏览器验证、DigiRL 最终 PDF 的工具大小限制；采用明确记录的 arXiv 版本。UI-R1 的 arXiv 与正式版数据规模不同，作为补充来源单列。GuidNav 部分汇总数字无法由分域表直接重建，因此本文不借其 headline 数字论证本项目预期增益。

来源清单记录版本和实际阅读范围；检索日志记录查询和访问限制。前文的本地实验结论来自项目记录，方案属于本文结合文献的推论，不是论文替本项目证明的结果。

**流程出处：** 本次用 literature-review 技能组织检索与记录；依该技能要求引用 Kassis 等，*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*（2026）。它是研究流程来源，不计入 GUI 方法论文，也不作为算法有效性的证据。[流程论文](https://arxiv.org/abs/2609.00065)

#### 9.10.3 经核对的来源索引

详细元数据与阅读范围见 [来源清单](gui_agent_sources.json)，查询与限制见 [检索范围](#scope)。以下链接固定到本次阅读的版本或出版页面。

<!-- VERIFIED_SOURCE_INDEX -->

本轮共 **28 篇核心文献 + 1 篇补充版本核验**；其中含基准、数据集、平台与理论，不是 28 种训练算法。流程引用另列，不计入上述数量。

| 编号 | 文献 | 本次版本 | 阅读用途 |
|---|---|---|---|
| 01 | [MobileGym: A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research](https://arxiv.org/html/2605.26114v1) | arXiv v1 | platform / simulation / evaluation |
| 02 | [Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents](https://arxiv.org/html/2504.00906v1) | arXiv v1 | agent_architecture / planning / grounding |
| 03 | [OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis](https://aclanthology.org/2025.acl-long.277.pdf) | ACL 2025 published paper | data_synthesis / trajectory_construction |
| 04 | [OpenMobile-2: Building Versatile Mobile Agents with Scalable Environments and App-Native Tools](https://arxiv.org/html/2610.14157v1) | arXiv v1 | mobile_agent / hybrid_tools / post_training |
| 05 | [WM-R1: Training GUI Agents to Reason and leverage World Models with Reinforcement Learning](https://arxiv.org/html/2608.27508v1) | arXiv v1 | world_model / reinforcement_learning |
| 06 | [Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) | ICML 1999 | theory / reward_shaping |
| 07 | [AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents](https://arxiv.org/html/2405.14573v5) | v5 | benchmark / evaluation |
| 08 | [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](https://arxiv.org/html/2404.07972v2) | v2 | benchmark / evaluation |
| 09 | [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/html/2307.13854v4) | v4 | benchmark / evaluation |
| 10 | [VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks](https://arxiv.org/html/2401.13649v2) | v2 | benchmark / evaluation |
| 11 | [On the Effects of Data Scale on UI Control Agents](https://arxiv.org/html/2406.03679v6) | v6 | dataset / offline_evaluation |
| 12 | [ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use](https://arxiv.org/html/2504.07981v1) | v1 | benchmark / grounding |
| 13 | [OSWorld 2.0: Benchmarking Computer Use Agents on Long-Horizon Real-World Tasks](https://arxiv.org/html/2606.29537v2) | v2 | benchmark / evaluation |
| 14 | [SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents](https://arxiv.org/html/2401.10935v2) | v2 | agent_method / supervised grounding continual pretraining and downstream adaptation |
| 15 | [Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents](https://arxiv.org/html/2410.05243v3) | v3 | agent_method / supervised visual grounding; separate planner in SeeAct-V |
| 16 | [ShowUI: One Vision-Language-Action Model for GUI Visual Agent](https://arxiv.org/html/2411.17465v1) | v1 | agent_method / offline instruction tuning and downstream supervised adaptation |
| 17 | [UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://arxiv.org/html/2501.12326v1) | v1 | agent_method / continual pretraining, annealing/SFT, reflective correction SFT and Agent DPO |
| 18 | [UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning](https://arxiv.org/html/2509.02544v2) | v2 | agent_method / CT/SFT/data iteration and multi-turn online PPO; critic/value pretraining |
| 19 | [InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners](https://arxiv.org/html/2504.14239v1) | v1 | agent_method / spatial reasoning distillation/SFT then RLOO |
| 20 | [GUI-R1 : A Generalist R1-Style Vision-Language Action Model For GUI Agents](https://arxiv.org/html/2504.10458v4) | v4 | agent_method / GRPO on sampled action responses with ground-truth rule rewards; separate supervised baseline |
| 21 | [ProgRM: Build Better GUI Agents with Progress Rewards](https://arxiv.org/html/2505.18121v1) | arXiv v1 | post_training / credit_assignment |
| 22 | [GUI-Shepherd: Reliable Process Reward and Verification for Long-Sequence GUI Tasks](https://arxiv.org/html/2509.23738v1) | arXiv v1 | post_training / credit_assignment |
| 23 | [Guiding VLM Agents with Process Rewards at Inference Time for GUI Navigation](https://arxiv.org/html/2504.16073v1) | arXiv v1 | post_training / credit_assignment |
| 24 | [Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/html/2505.10978v3) | arXiv v3 | post_training / credit_assignment |
| 25 | [AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress](https://arxiv.org/html/2511.08325v1) | arXiv v1 | post_training / credit_assignment |
| 26 | [WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning](https://arxiv.org/html/2411.02337v3) | arXiv v3 | post_training / credit_assignment |
| 27 | [DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning](https://arxiv.org/html/2406.11896v1) | arXiv v1 | post_training / credit_assignment |
| 28 | [MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents](https://arxiv.org/html/2509.18119v2) | arXiv v2 | post_training / credit_assignment |
| 29（补） | [UI-R1: Enhancing Efficient Action Prediction of GUI Agents by Reinforcement Learning](https://arxiv.org/html/2503.21620v5) | arXiv v5 (2025-05-24), AAAI2026 publication metadata and PDF opening | supplementary / single_step_rl / version_check |
