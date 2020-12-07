# model-based-social-navigation
Code for **Learning World Transition Model for Socially Aware Robot Navigation** in ICRA2021 is coming soon.



Paper is available [here](https://arxiv.org/abs/2011.03922).



Video is available [here](https://www.youtube.com/watch?v=K7cBViQ9Vds&t=11s).



# Get ready
```
catkin_make -j6
source devel/setup.bash
```

# MODEL-FREE
```
FOUR-AGENT
roslaunch model_free_version start.launch
```

- 主要对应的文件有
- *main.py* 4 agent model-free主函数
- *policy.py* 4 agent model-free主函数
- *environment_four.py* 4 agent真实环境
- *agent.py* agent函数，包含获取状态以及reward以及控制函数等
- *utils.py* 真实环境数据存储池



# MODEL-BASED
```
roslaunch model-based-social-navigation start.launch
```

- 主要对应的文件有
- *main_mbpo.py* 1/4 agent主函数
- *env_sample.py* 真实环境接口
- *env_sample_four.py* 真实环境接口(4)
- *env_predict.py* 虚拟环境接口
- *environment_one_agent.py* 真实环境
- *environment_four_agent.py* 真实环境(4)
- *agent_mbpo.py* agent函数，包含获取状态以及reward以及控制函数等
- *replay_buffer_env.py* 真实环境数据存储池
- *replay_buffer_model.py* 虚拟环境数据存储池
- *policy.py* 策略
- *transition_model.py* 预测模型
- *ensemble_model_train_mcnet_all.py* 预测模型接口
