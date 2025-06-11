scripts/vis/vismotion_mj.py
用胶囊体代替骨骼肢体，将amass数据集中的动作片段进行可视化（可暂停、播放下一个动作等），若要可视化多个动作片段，需要设置num_motions与motion_file，motion_file可为单个pkl文件或一个包含多个pkl文件的文件夹

motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
这句代码中，limb_weights=[np.zeros(10)] * num_motions这种用法，会导致num_motions个运动片段所对应的[np.zeros(10)]共同指向同一个numpy数组对象，因此修改其中任意一个元素，所有元素都会改变

mj_data.qpos[3:7] = root_rot[0].cpu().numpy()[[3, 0, 1, 2]]
注意mujoco中表示旋转的四元数与smpl模型中表示旋转的四元数的w,x,y,z的顺序
