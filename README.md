# model-based-social-navigation
Code for **Learning World Transition Model for Socially Aware Robot Navigation** in ICRA2021 is coming soon.

[![arXiv](https://img.shields.io/badge/arxiv-2011.03922-B31B1B.svg)](https://arxiv.org/abs/2011.03922)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![video](https://img.shields.io/badge/video-icra2021-blue.svg)](https://www.youtube.com/watch?v=K7cBViQ9Vds&t=11s)



Paper is available [here](https://arxiv.org/abs/2011.03922).


Video is available [here](https://www.youtube.com/watch?v=K7cBViQ9Vds&t=11s).



# Get ready
```
git clone https://github.com/YuxiangCui/model-based-social-navigation.git
catkin_make -j6
source devel/setup.bash
```

# MODEL-FREE
```
roslaunch model_free_version start.launch (FOUR-AGENT)
```

- *main.py* (4 agent main function)
- *policy.py* (policy network)
- *environment_four.py* (4 agent environment)
- *agent.py* (agent's states, reward, action...)
- *utils.py* (replay buffer)



# MODEL-BASED
```
roslaunch model-based-social-navigation start.launch
```

- *main_mbpo.py* (1/4 agent main function)



- *env_sample.py*
- *environment_one_agent.py* (real 1 agent environment)
- *env_sample_four.py* 
- *environment_four_agent.py* (real 4 agent environment)



- *agent.py* (agent's states, reward, action...)
- *replay_buffer_env.py* (real data replay buffer)
- *replay_buffer_model.py* (virtual data replay buffer)
- *policy.py* (policy network)



- *transition_model.py* (world transition model)
- *ensemble_model_train_mcnet_all.py* 
- *env_predict.py* (virtual environment)
