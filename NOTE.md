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