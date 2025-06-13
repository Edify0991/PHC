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