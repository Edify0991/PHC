scripts/vis/vismotion_mj.py
用胶囊体代替骨骼肢体，将amass数据集中的动作片段进行可视化（可暂停、播放下一个动作等），若要可视化多个动作片段，需要设置num_motions与motion_file，motion_file可为单个pkl文件或一个包含多个pkl文件的文件夹

motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
这句代码中，limb_weights=[np.zeros(10)] * num_motions这种用法，会导致num_motions个运动片段所对应的[np.zeros(10)]共同指向同一个numpy数组对象，因此修改其中任意一个元素，所有元素都会改变

mj_data.qpos[3:7] = root_rot[0].cpu().numpy()[[3, 0, 1, 2]]
注意mujoco中表示旋转的四元数与smpl模型中表示旋转的四元数的w,x,y,z的顺序

### scripts/data_process/fit_smpl_shape.py

运行时会加载phc/data/cfg中的config.yaml作为基参数文件，运行该代码时利用hydra修改config.yaml中的robot参数，robot为需要重定向的目标机器人，此时会指向phc/data/cfg/robot中与${robot}相应yaml参数文件

smpl_pose_modifier参数应该是将“T”字形smpl模型与机器人对齐，若用G1举例，则R_Shoulder_Pitch转动pi/2后可与机器人对齐，但是这里修正的旋转角度不知道是以何坐标系为参考？大致意思是肩关节从胳膊水平打开变为竖直放下，肘关节弯曲90°，但细节还需要打磨！！！

### vis/vis_q_mj.py
目前只能可视化一个重定向后的动作片段，希望能修改成同时重定向多个动作片段

当motion_number超出最大动作数量后，会报错，修复这个错误

### vis/vis_motion.py
在修改生成SMPL模型对应的MJCF格式模型文件test_good的路径时，有意将/tmp由根目录修改为项目目录tmp/，但此时smpl_sim包下的smpl_local_robot.py中用来生成目标xml文件的代码仍将路径设置为/tmp，所以这里也要进行相应修改，否则其会将MJCF格式模型对应的mesh文件生成到/tmp中，而在tmp中的xml文件将会找不到mesh文件，因此会报错

### utils/torch_humanoid_batch.py
self.actuated_joints_idx = np.array([self.body_names.index(k) for k, v in mjcf_data['body_to_joint'].items()])

mjcf_data['body_to_joint'] 是一个字典，key 是 body 名字，value 是对应 joint 名字。
这行代码的作用是：遍历所有有 joint 的 body，把它们在 body_names 列表中的下标收集起来，方便后续只对这些节点做动力学/控制。

如果既没有 <freejoint/>，也没有 <joint type="free"/>，就默认前6个 joint 是根节点的自由度，从第7个 joint 开始才是实际的运动关节（比如髋、膝、踝等）。
所以 dof_axis 只统计从第7个 joint 开始的轴向信息，前6个 joint 被当作根自由度跳过

那之前的报错问题可能确实如你所说，我传入的rotations是关于所有joint的，因为我在fit_smpl_motion.py中需要计算通过fk_batch计算得到的各个关节世界位置与数据集中的关节世界位置之差，进而进行迭代，但是我的关节数少于body数目，而要计算正运动学需要通过body来遍历到所有关节，但运算时就会因为rotations的维度问题导致出错，此外在计算结束后返回的是所有body的世界坐标与姿态，但其实fit_smpl_motion.py中需要的是关节的世界坐标，所以还需要一步筛选，我该如何进行修改

我现在不想像你之前说的那样，将rotation扩充到与body数量相同，我现在希望利用actuated_joints_idx这个变量，因为它代表每个关节所对应的body的索引号，这样的话，我希望在for i in range(J):
            print(f"i={i}, parent={self._parents[i]}, rotations_world_len={len(rotations_world)}")
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                # print(f"rotations_world[{i}] shape:", rotations_world[self._parents[i]].shape)
                print(f"rotations.shape: {rotations.shape}, i: {i}, (i-1):i = {(i-1, i)}")
                print(f"rotations[:, :, (i-1):i, :].shape: {rotations[:, :, (i-1):i, :].shape}")
                
                jpos = (torch.matmul(rotations_world[self._parents[i]][:, :, 0], expanded_offsets[:, :, i, :, None]).squeeze(-1) + positions_world[self._parents[i]])
                rot_mat = torch.matmul(rotations_world[self._parents[i]], torch.matmul(self._local_rotation_mat[:,  (i):(i + 1)], rotations[:, :, (i - 1):i, :]))
                # rot_mat = torch.matmul(rotations_world[self._parents[i]], rotations[:, :, (i - 1):i, :])
                # print(rotations[:, :, (i - 1):i, :].shape, self._local_rotation_mat.shape)
                
                positions_world.append(jpos)
                rotations_world.append(rot_mat)这个循环中进行修改，当遍历到actuated_joints_idx中所指向的body时，说明该body含有joint，此时就可以在求jpos和rot_mat时，当遇到含有joint的body，就使用rotations中的对应元素，否则遇到不含有joint的body，就使用单位旋转矩阵，而在执行到wbody_pos, wbody_mat = self.forward_kinematics_batch(pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans)这一步以后，再利用actuated_joints_idx将其中索引对应的位置向量和旋转矩阵赋值给wbody_pos和wbody_mat，我该如何写代码？

由于kepler模型中存在body与joint数目不等的情况，不像宇树模型中没有joint的连杆只是用geom标签不使用body标签，这样在解析xml模型时就不会出错，为此修改了运动重定向时关于机器人正运动学的代码，重定向流程基本走通，不过效果感觉不是太好，下一步继续修正改善重定向结果，搞明白vis_j_mj.py代码中红色小球对应什么，该如何调整才能使其与机器人本体对应的更好，此外还要尝试解决机器人悬空的问题，还有就是参考human2humanoid中重定向相关代码，将重定向效果在isaacgym或isaaclab中可视化，在这个过程中要基本搞清对smpl模型的操作。还要同步进行训练部分的代码学习，尤其是PHC中PNN的用法与作用，目前还没搞懂PHC论文的主要工作是什么，是否可以一个策略学习绝大部分动作。

### run_hydra.py

#### RLGPUAlgoObserver
* 继承于AlgoObserver类，用于监控和记录训练过程中的统计信息，特别是成功率相关的指标
* after_init：初始化统计工具和 TensorBoard writer，创建了一个滑动平均计算器
* process_infos：处理环境返回的信息，更新成功率统计
* after_clear_stats：清除统计数据
* after_print_stats: 将统计结果写入 TensorBoard，记录训练进度


### humanoid_amp.py
* fetch_amp_obs_demo：为判别器(discriminator)创建参考动作的AMP观测数据，从动作库中采样参考动作，并生成对应的AMP观测数据，供判别器训练使用

### motion_lib_base.py
* sample_motions：从动作库中随机采样 n 个动作序列的索引
* load_motions:
  * 记录单个骨架的关节数和环境数量（骨架数量）
  * 通过随机采样或顺序采样选择动作序列，随机采样的采样概率可自适应修改，顺序采样会设置起始动作索引与取模运算（环境数量与动作数量不匹配）。随机采样适用于训练模式，可增加数据多样性，修改采样概率后可重点训练困难的动作；顺序采样用于评估/测试模式，可确保所有动作都被覆盖，便于调试和结果复现。
  * 设置当前批次的动作相关信息和批次内采样概率。将动作ID转换为one-hot编码，获取当前批次动作序列的字符串标识符，计算批次内采样概率；  
    实现了两级采样机制：第一级: 从所有动作中采样一批动作给各个环境；第二级: 从当前批次的动作中采样参考数据给判别器
  * 根据采样的索引获取对应的动作数据，设置每个进程只是用一个线程，准备多进程加载和处理动作数据
  * 建一个空字典用于累积多进程的处理结果，字典能确保结果的顺序性，计算每个进程应该处理的任务数量
  * 实现主-从并行处理模式加载动作数据，主进程处理第一批任务 + 协调整体流程，子进程并行处理其余任务批次，通过队列传递结果，字典累积最终数据
  * 处理多进程加载的动作数据结果的核心部分，提取动作基本信息（动作序列的帧率、每帧时间间隔、动作序列总帧数、动作序列总时长），处理SMPL相关数据（轴角表示的姿态数据与性别和身体形态参数），如果启用真实轨迹模式，存储Quest VR头显的运动数据（位置、旋转、角速度、线速度）
  * 将所有处理好的动作数据整合成统一的张量格式，方便后续的快速访问和采样，计算每个动作序列在大张量中的起始索引
* load_motion_with_skeleton：需要在不同子类中实现，使用不同的机器人需要不同的实现，将原始动作数据（AMASS或其他格式的动作数据）转换为可用于训练的骨架运动SkeletonMotion数据
  * 给每个进程设置独一无二的随机种子，保证多进程数据处理的随机性，不同进程处理相同数据时会产生不同的增强效果从而增加数据多样性，避免所有进程产生相同的训练样本可以提高训练质量
  * 进行数据加载和预处理，这时可能会碰到 之前预处理时处理文件夹情况时存储的内容为文件名而不是具体动作数据的情况，这里进行动作数据读取
  * 如果动作序列太长，会随机选择一个子序列
  * 若在训练模式下（不能在test/eval模式下），进行数据增强，随即旋转整个动作序列的朝向（测试时保持原始朝向）
  * 使用SMPL网格计算准确的地面接触点，调整根部位置，确保角色脚部接触地面，避免角色悬浮或陷入地面
  * 进行骨架状态构建，将姿态和位置数据转换为骨架状态对象，创建SkeletonMotion对象，包含完整的运动学信息
  * Quest VR数据处理

* load_data：从文件或文件夹中加载动作数据，并根据不同的筛选条件进行处理，为后续的动作库初始化做准备
  * 文件模式：_motion_data_list 内容是实际的动作数据对象数组，_motion_data_keys 内容是动作序列的名称数组
  * 目录模式：_motion_data_list 和_motion_data_keys 内容都是文件路径字符串数组

### base_task.py
* 是强化学习环境的核心组件，它封装了与 Isaac Gym 物理仿真器的交互，为强化学习训练提供标准化的环境接口
* init：环境初始化，初始化强化学习环境的所有核心组件，包括显示管理（设置虚拟显示器（用于无头模式下的渲染））、设备配置、缓冲区分配（为 RL 训练分配关键的张量缓冲区——观测、状态、奖励、重置标志、progress）
* create_viewer：创建可视化界面，可实时观察智能体的学习进度、通过键盘快捷键控制训练过程、录制训练视频和状态数据
* step：RL环境的主要步进函数，先进行动作预处理，应用域随机化噪声（提高泛化能力），然后物理步进，执行动作并推进物理仿真，最后状态计算，计算新的观测、奖励、重置标志
* pre_physics_step：动作应用，将智能体的动作应用到仿真环境中，在子类中实现具体的动作到仿真控制的映射
* post_physics_step：状态更新，物理仿真步进后计算 RL 所需的信息（计算观测、奖励，判断终止条件，更新进度计数）
* _physics_step：物理仿真循环，控制**策略更新**频率与**物理仿真**频率的比例，在物理步进中集成可视化渲染
* apply_randomizations：域随机化
* _record_states：记录训练状态
* _clear_recorded_states: 清除记录
* _write_states_to_file: 保存状态到文件
* setup_talk_client与talk：用于WebSocket通信，分布式训练支持 远程监控（通过网络监控训练进度）、远程控制（远程重置环境、开始/停止录制）、多用户协作（支持多人同时观察训练过程）
  * 在线程内创建新的独立异步事件循环

### humanoid.py
* 是专门针对人形机器人强化学习任务的核心基类，为AMP、PHC等算法提供了人形机器人特有的功能，相较于BaseTask，专门针对人形机器人的物理仿真和观测计算，其子类(如HumanoidAMP)实现具体的学习算法(AMP、PHC等)
* init：初始化人形机器人强化学习环境的核心组件，加载机器人配置（支持SMPL人体模型和真实机器人(H1、G1)）、设置控制模式（PD控制、力控制等）、配置观测和动作空间（根据机器人类型自动计算维度）、初始化张量缓冲区（为并行训练分配GPU内存）
* load_humanoid_configs：配置加载分发器，根据机器人类型分发到不同的配置加载函数
* load_smpl_configs： SMPL人体模型配置，每个关节3自由度(轴角表示)
* load_robot_configs：真实机器人配置，每个关节1自由度(关节角)
* create_sim：仿真世界创建，创建Isaac Gym仿真环境，设置坐标系为Z轴向上，创建地面（配置摩擦力、弹性等物理参数），创建环境实例（支持数千个并行环境）
* _create_envs：批量环境创建
  * SMPL情况下可多样化资产生成（生成不同的SMPL XML文件），多进程资产创建（利用多核CPU加速XML文件生成）与物理属性设置（质量、惯性、摩擦力等）
* _build_env：单个环境构建，配置自碰撞过滤以及PD控制参数
* _compute_humanoid_obs：人形机器人观测计算，这是与BaseTask最大的区别之一，专门计算人形机器人的复杂观测，要将身体状态转换到根坐标系后计算出相对观测
  * SMPL情况下计算观测维度时，分解为根部高度（1）、身体位置（相对，3）、身体旋转（6D表示，6）、身体速度（3）、身体角速度（3）、去除根部位置（因为已用相对位置，-3）
  * 维护历史状态缓冲区（观测维度=基础观测x(历史步数+1)）
* pre_physics_step：动作预处理，与BaseTask的区别是，
BaseTask面向通用动作处理，Humanoid针对人形机器人的PD控制、力矩控制
  * SMPL：要将轴角动作转PD目标
  * 真实机器人：直接关节角目标
* _action_to_pd_targets：将归一化的网络输出转换为关节目标位置
  * SMPL特殊处理：加强膝关节控制
* _physics_step：物理步进,控制频率管理
* _compute_torques：力矩计算 (H1/G1机器人)
* reset：环境重置，若开启“安全重置”机制，会再执行一步仿真然后再次重置，为了消除任何残留的物理状态异常
* _reset_actors：角色状态重置
* _create_smpl_humanoid_xml：SMPL XML生成，支持多进程XML生成
* sample_char_color：用于角色着色，不同环境使用不同颜色，便于观察训练过程
* _build_key_body_ids_tensor：关键身体部位索引，用来快速访问重要身体部位（如脚、手）的物理状态

### humanoid_amp.py
* 是实现AMP（Adversarial Motion Prior）算法的核心组件，它继承自 Humanoid 类，为动作模仿学习提供了完整的基础设施
* 实现了：
  * 数据管理: 加载和管理大量人类动作数据
  * 观测计算: 构建适合判别器训练的观测表示
  * 状态初始化: 从参考动作中初始化环境状态
  * 性能优化: 缓存机制和JIT编译提高效率
  * 调试支持: 丰富的调试和测试工具
* init：AMP环境初始化，涉及到关键AMP配置（状态初始化策略、AMP观测历史步数、是否包含根部高度观测）  
  与Humanoid类相比，主要区别在于：增加了AMP特有的观测缓冲区、配置了动作参考数据的初始化策略、设置了判别器需要的历史观测
  * StateInit 枚举类 - 初始化策略（默认初始姿态、从动作序列开始、随机时间点初始化、混合策略）
* _load_motion：动作数据加载，加载人类动作捕捉数据作为AMP的参考，数据来源于AMASS数据集中的人类动作，并经过fit_smpl_motion.py 处理转换为机器人格式
* resample_motions：重新采样动作，在训练过程中重新加载动作数据，增加训练多样性
* _setup_character_props：设置AMP观测维度
* _compute_amp_observations：计算AMP观测，计算用于判别器训练的观测数据
* _update_hist_amp_obs：更新历史观测，维护多步历史观测，为判别器提供时序信息
* fetch_amp_obs_demo：获取参考观测数据，也是AMP训练核心函数，首先随机采样动作ID和时间，然后构建参考观测，最后返回给判别器训练使用
* build_amp_obs_demo：构建参考观测，从动作库中构建多步历史观测
* _reset_actors：智能体重置策略，根据配置选择重置策略
* _reset_ref_state_init：参考动作初始化，AMP的核心重置策略——先采样参考状态，设置环境状态为参考动作的状态，还要记录初始化信息
* _sample_ref_state：采样参考状态，从动作库中随机采样参考状态用于环境初始化
* _get_fixed_smpl_state_from_motionlib：SMPL高度修正，解决SMPL模型的地面接触问题
* _get_state_from_motionlib_cache：动作状态缓存，缓存相同查询的结果，避免重复计算
* build_amp_observations_smpl：是AMP观测构建函数（JIT编译）
* _hack_motion_sync：动作同步测试，强制环境跟随参考动作，用于验证动作数据的正确性
* _hack_output_motion：动作输出，记录和输出训练过程中的动作序列


### humanoid_amp_task.py
* 数据流：AMASS数据 → fit_smpl_motion.py → 机器人动作数据 → MotionLib → HumanoidAMP → HumanoidAMPTask → HumanoidIm
* init：任务导向AMP环境初始化
  * AMP算法层面：
    * 继承动作库: 从 fit_smpl_motion.py 处理的数据中加载人类动作参考
    * 保持判别器功能: 维护AMP的对抗性训练能力
    * 动作先验: 利用大量AMASS数据学习自然动作模式
  * PHC算法扩展:
    * 任务观测开关: _enable_task_obs 控制是否添加任务特定观测
    * 任务标识: self.has_task = True 标记此环境包含具体任务目标
* get_obs_size：动态观测维度计算，所继承父类方法计算的是AMP基础观测维度，其子类（HumanoidIm计算任务观测）
* pre_physics_step：物理步进前的统一处理，从父类方法执行AMP动作处理，然后进行任务状态更新。
  * 任务更新具体内容：
    * 目标跟踪: 更新目标位置和朝向
    * 进度计算: 计算任务完成度
    * 动态目标: 处理移动目标或动态任务
    * 约束检查: 验证任务约束条件
* render： 分层渲染系统，并可进行任务可视化（子类中实现）
  * 可视化层次结构：
    * AMP层可视化：
      - 人形机器人模型 (基于kepler.xml/g1.xml)
      - 参考动作轨迹 (来自fit_smpl_motion.py处理的数据)
      - 判别器状态指示
      - AMP观测信息
    * 任务层可视化：
      - 目标点标记
      - 路径规划线
      - 任务进度条
      - 完成状态指示器
* _update_task：任务状态更新接口
* _reset_envs：分层环境重置（AMP重置和任务重置）
* _compute_observations：组合观测计算核心
* _compute_task_obs：任务观测计算接口
* _compute_reward：奖励计算接口
* _draw_task：任务可视化接口

### humanoid_im.py
* 继承自 HumanoidAMPTask，实现了完整的动作模仿（Imitation）功能
* init：关于PHC算法的关键配置，是否使用全身体奖励计算、是否启用未来轨迹跟踪（PHC的和新特性）、未来轨迹采样数量（用于Progressive Control）
  * 跟踪身体部位，若是VR，则只有跟踪的三个关键点（头+双手），否则为motion tracking所需的所有关节
* _load_motion：动作数据加载
  * 加载由 fit_smpl_motion.py 处理后的PKL文件
  * 支持SMPL和真实机器人(H1/G1)两种数据格式
  * 使用 torch_humanoid_batch.py 进行前向运动学计算
  * 若新添加机器人，需要编写motionlib
* resample_motions：动作重采样
* get_task_obs_size：任务观测维度计算
  * 位置差异: 3维 (x, y, z)
  * 旋转差异: 6维 (6D旋转表示)
  * 速度差异: 3维
  * 角速度差异: 3维
  * 总计: 15维/身体部位
* _compute_task_obs：PHC核心观测计算
  * Multiplicative Progressive Control准备:
  * 时间步采样: 为每个环境采样多个未来时间点
  * 状态预测: 从动作库中获取未来状态作为控制目标
  * 观测构建: 将当前状态与未来目标状态的差异作为观测
* _compute_reward：分层奖励计算
  * 距离敏感的奖励策略，远距离采用位置奖励，近距离采用模仿奖励，此外还有标准全身模仿奖励
    * 位置奖励: 引导智能体到达目标区域  
    * 模仿奖励: 确保动作自然性
  * 功率奖励: 能效优化
* _get_state_from_motionlib_cache：状态缓存和预测，是Progressive Control核心实现
  * 缓存机制避免重复计算
  * 为Progressive Control服务:
    * 状态预测: 获取未来时间点的参考状态
    * 性能优化: 缓存避免重复的前向运动学计算
    * 偏移处理: 支持全局位置偏移，实现灵活的任务目标
* _reset_ref_state_init：智能重置策略
  * 随机距离初始化，支持Progressive Control
* compute_imitation_observations_v6：JIT编译的观测计算函数，高效观测计算
  * JIT编译: 提高计算效率，支持大规模并行训练
  * 局部坐标系: 确保观测的旋转不变性
  * 批量处理: 同时处理多个时间步和环境
* create_o3d_viewer：3D可视化
* render：实时渲染，渲染当前状态和参考状态，同时显示当前和参考动作
* _compute_reset：智能重置逻辑，支持动作循环，用于长期训练，会更新全局偏移以保持连续性
* Multiplicative Progressive Control的实现核心思想（如何实现渐进式控制）：
  * 多时间步预测: _fut_tracks 启用时，计算多个未来时间点的目标状态（未来轨迹采样）
  * 距离分层控制: zero_out_far 根据距离目标的远近采用不同控制策略
    * 远距离：方向控制
    * 中距离：位置控制
    * 近距离：精确模仿
  * 动态目标更新: 通过 _global_offset 实现目标位置的动态调整

### humanoid_im_mcp.py
* 是PHC算法框架中的高级实现，它实现了MCP（Mixture of Control Primitives）——多控制基元混合的概念
* MCP（Mixture of Control Primitives）算法思想：
  * 将复杂的人形机器人控制分解为多个专门的控制基元（primitives）
  * 每个基元负责特定类型的动作（如走路、跑步、转向等）
  * 通过权重混合不同基元的输出，实现流畅的动作转换
* init：MCP环境初始化，MCP核心配置
  * 控制基元数量
  * 混合策略：离散混合（硬切换，某一时刻只使用一个基元） vs 连续混合（软混合：同时使用多个基元）
  * 网络架构：是否使用PNN，以及是否启用侧向连接（基元间信息交换）
    * PNN架构特点：
      * 渐进式学习: 每个基元逐步学习，保留之前的知识
      * 侧向连接: has_lateral=True 时，基元间可以共享信息
      * 防止遗忘: 新基元训练时不会破坏已学习的基元
* _setup_character_props：动作维度重定义，**动作维度变为基元数量**
  * 标准PHC: num_actions = num_dof（关节数，如kepler的28个关节）
  * MCP: num_actions = num_prim（基元数，如3个基元）
  * 意义：
    * 传统控制：直接输出关节动作
    * MCP控制：输出基元权重
* get_task_obs_size_detail：观测信息扩展
  * 为上层控制器提供基元数量信息，用于权重维度设置
* step：MCP核心执行函数
  * 观测归一化，使用训练时的统计信息归一化观测，obs_buf 来自HumanoidIm的复合观测计算
    * AMP观测：基于fit_smpl_motion.py处理的AMASS数据
    * PHC任务观测：目标位置、进度、未来轨迹等
    * 机器人状态：基于kepler_fitting.yaml/unitree_g1_fitting.yaml的关节映射
  * 权重处理策略：离散模式——选择权重最大的基元（硬切换）；连续模式——连续混合（平滑过渡）
  * 基元动作生成和混合：PNN模式——渐进式神经网络；标准模式——独立的actor网络
    * 对于每个环境i和每个关节j：  
      action[i,j] = Σ(k=0 to num_prim-1) weights[i,k] * primitive_action[i,k,j]
  * MLP旁路（可选）：完全绕过基元混合，直接使用传统MLP
  * 标准PHC执行流程：
    * 应用混合后的动作到机器人
    * 物理仿真步进
    * 计算奖励和下一步观测

### common_agent.py
* 核心作用：
  * 强化学习训练引擎：基于PPO算法实现策略优化
  * 数据流管理：处理从环境到网络的数据传递
  * 模型管理：网络初始化、保存、加载
  * 训练监控：性能指标记录和可视化
* init：强化学习智能体初始化
  * 配置加载和网络构建
  * 观测归一化设置
    * 处理来自 HumanoidIm 的复合观测（AMP观测+任务观测）
    * 确保不同尺度的观测（位置、角度、速度）在相同范围内
* init_tensors：经验缓冲区初始化
  * 经验回放：为PPO算法准备存储空间
  * 时序数据：存储(s, a, r, s')四元组用于策略优化
* train：主训练循环
  * 数据收集阶段
  * 策略优化阶段
  * 性能监控
* play_steps：环境交互数据收集
  * 观测获取：obs包含来自HumanoidIm的复合观测（AMP观测+任务观测）与环境内部状态
  * 动作执行
  * 奖励计算
* get_action_values：策略网络前向传播
  * 网络输出解析包含：动作输出、状态价值函数、动作对数概率、动作均值（连续动作）、动作标准差
  * calc_gradients：PPO损失计算和反向传播
    * 计算Actor损失（策略梯度）
    * 计算Critic损失（价值函数）
    * 计算边界损失（动作约束）
* _actor_loss：Actor损失（策略优化）
* _critic_loss：Critic损失（价值函数优化）
* discount_values：GAE优势函数计算
  * 优势函数：A(s,a) = Q(s,a) - V(s)
  * 减少方差：通过λ参数平衡偏差和方差
  * 时序差分：δ = r + γV(s') - V(s)
* prepare_dataset：训练数据准备
* _setup_action_space：动作空间设置
  * 对于MCP：actions_num = num_primitives
  * 对于标准PHC：actions_num = num_dof
* bound_loss：动作边界约束
  * 防止动作超出机器人关节限制
* 多层奖励整合：
  * AMP自然性奖励
  * 模仿学习奖励  
  * 任务完成奖励
  * 进度奖励
* _log_train_info：训练指标记录
  * 监控指标：
    * 性能指标：FPS、训练时间、内存使用
    * 学习指标：损失值、KL散度、梯度范数
    * 任务指标：奖励、episode长度、成功率

### common_player.py
* AMASS数据 → fit_smpl_motion.py → torch_humanoid_batch.py → HumanoidIm环境 → CommonAgent(训练) → 训练完成的模型 → CommonPlayer(推理/评估) → 实际控制机器人
* 核心作用：
  * 推理执行器：使用训练好的PHC模型进行实际推理和控制
  * 评估工具：评估训练模型的性能表现
  * 演示系统：可视化展示PHC算法的控制效果
  * 部署桥梁：连接训练阶段和实际应用
* init：推理器初始化
  * 与训练器的区别：
    * 训练阶段 (CommonAgent)：focus on 学习和优化策略
    * 推理阶段 (CommonPlayer)：focus on 使用已训练的策略
* run：主推理循环
* get_action：策略网络推理
  * PHC策略网络推理过程：
    * 观测预处理：
      * 归一化处理（使用训练时的统计信息）
      * 观测包含多个组件（来自fit_smpl_motion.py的AMP观测、PHC任务相关观测、历史观测）
    * 网络前向传播：
      * 确定性推理：直接使用均值
      * 随机推理：从分布中采样
    * 动作后处理：
      * 动作裁剪到有效范围
* env_step：环境交互
  * 动作传递到HumanoidIm：env.step(actions) -> HumanoidIm.step() -> 物理仿真 -> 奖励计算 -> 下一步观测
* _build_net：网络构建
  * 网络配置适配：根据机器人类型（自由度不同）、任务类型进行调整
* env_reset：环境重置
  * env.reset() -> 从动作库采样初始状态 -> 设置机器人姿态（基于fit_smpl_motion.py数据） -> 重置任务目标 -> 计算初始观测
* _setup_action_space： 动作空间设置
* 与训练阶段的对比
  * 训练阶段 (CommonAgent)：
    * 目标：优化策略参数
    * 模式：model.train()
    * 梯度：需要计算和反向传播
    * 探索：包含噪声的动作采样
    * 数据：收集训练数据
  * 推理阶段 (CommonPlayer)：
    * 目标：执行和评估策略
    * 模式：model.eval()
    * 梯度：torch.no_grad()
    * 确定性：通常使用确定性动作
    * 应用：实际控制和性能评估

### amp_agent.py
* AMASS数据 → fit_smpl_motion.py → torch_humanoid_batch.py → HumanoidIm环境 → AMPAgent (AMP+PPO训练) → 策略网络+判别器 → 自然动作控制
* 核心作用：
  * AMP算法实现：结合判别器的对抗性模仿学习
  * PPO扩展：在PPO基础上添加判别器损失和AMP奖励
  * 数据整合：处理来自 fit_smpl_motion.py 的专家数据
* init：AMP智能体初始化
  * AMP特有的归一化与判别器奖励归一化
  * 模型冻结机制，按照PHC的渐进式训练（先冻结基础模型），冻结归一化参数
* _build_amp_buffers：AMP缓冲区构建
  * 存储AMP观测
  * 专家数据缓冲区（来自fit_smpl_motion.py）-_amp_obs_demo_buffer
  * 回放缓冲区（存储智能体生成的数据）-replay_buffer_size
* play_steps：AMP数据收集
  * 标准PPO数据收集+AMP特有（收集AMP观测）
* train_epoch：AMP训练主循环
  * 数据收集（play_steps）
  * AMP专家数据更新（_update_amp_demos）
  * 准备AMP数据
  * 回放数据
  * 三类AMP数据的作用：
    * 当前智能体生成的数据：amp_obs
    * 来自fit_smpl_motion.py的专家数据：amp_obs_demo
    * 历史智能体数据（稳定训练）：amp_obs_replay
* calc_gradients：AMP核心损失计算
  * 预处理观测数据与AMP数据
  * 网络前向传播：PPO组件与AMP判别器输出（三类AMP数据的判别结果）
  * 损失计算：PPO损失 AMP判别器损失 熵损失 边界损失
* _disc_loss：AMP判别器损失核心
  * 二元分类损失：智能体数据标记为0（假），专家数据标记为1（真）
  * WGAN-GP梯度惩罚：梯度惩罚（确保Lipschitz约束）
  * 正则化项：Logit正则化与权重衰减
* _calc_amp_rewards：AMP奖励计算
  * 负对数似然奖励
  * 奖励归一化
  * AMP奖励含义：
    * 高奖励：智能体动作接近专家数据（来自fit_smpl_motion.py）
    * 低奖励：智能体动作偏离专家数据
    * 自动调节：通过判别器动态调整奖励信号
* _combine_rewards：多层奖励组合（任务奖励+AMP风格奖励）
* pre_epoch/post_epoch：训练周期管理
  * 动态调整权重，渐进式训练调度
  * SMPL形状重采样（对应不同的人体形状），重新加载fit_smpl_motion.py的数据
  * 冻结归一化参数（确保训练稳定性）
* _update_amp_demos：专家数据管理
  * 从环境获取专家数据（基于fit_smpl_motion.py处理的AMASS数据）-_fetch_amp_obs_demo
* freeze_state_weights/unfreeze_state_weights：状态管理函数
* _preproc_obs：观测预处理
  * 温度参数用途：防止训练过程中归一化参数的更新影响梯度计算的一致性
* _store_replay_amp_obs：回放缓冲区管理
  * 随机保留策略（避免缓冲区溢出）
* _assemble_train_info：训练指标记录
  * AMP特有指标disc
  * reward_raw
  * sym_loss