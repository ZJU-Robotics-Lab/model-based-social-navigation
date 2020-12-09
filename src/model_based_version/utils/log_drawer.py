from tensorboard.backend.event_processing import event_accumulator
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def parse_data(records, name, start_step, end_step, step_interval):
    x = []
    y = []
    for record in records:
        ea = event_accumulator.EventAccumulator(record)
        ea.Reload()
        average_reward = ea.scalars.Items('Reward/average_reward')
        # print(len(average_reward))
        first_reward = True
        for reward in average_reward:
            # print(reward.step)
            if reward.step <= start_step and reward.step >= start_step - 5:
                previous_sum = reward.step * reward.value
            elif reward.step >= start_step and reward.step < end_step:# and reward.step % step_interval == 1:
                current_reward = max((reward.step * reward.value - previous_sum)/(reward.step - start_step), -1)
                if first_reward:
                    first_reward = False
                    for i in range(start_step, reward.step):
                        x.append(i-start_step)
                        y.append(current_reward)
                if len(x) > 0 and reward.step - start_step > x[-1]:
                    for i in range(x[-1]+start_step+1, reward.step):
                        x.append(i - start_step)
                        y.append(y[-1])
                x.append(reward.step - start_step)
                y.append(current_reward)
        for i in range(x[-1]+1, end_step - start_step):
            x.append(i)
            y.append(y[-1])
        print(len(x), len(y))
        print(x[-20:])
        print(y[-20:])
    
    data = pd.DataFrame({
        'step': x,
        'average reward': y,
        'Methods': name
    })
    return data


if __name__ == '__main__':
    single_agent = True
    if not single_agent:
        # parameters
        start_step = 3000
        end_step = 10000
        step_interval = 20

        # model_free_four_20
        model_free_four_20 = [
            'xxx/events.out.tfevents.xxx.xxx',
        ]
        data1 = parse_data(model_free_four_20, 'Ours_w/o_20', start_step, end_step, step_interval)

        # model_free_four
        model_free_four = [
            'xxx/events.out.tfevents.xxx.xxx',
        ]
        data2 = parse_data(model_free_four, 'Ours_w/o_1', start_step, end_step, step_interval)
        
        # fully_distributed_four
        fully_distributed_four = [
            'xxx/events.out.tfevents.xxx.xxx',
        ]
        data3 = parse_data(fully_distributed_four, 'FDMCA', start_step, end_step, step_interval)

        # mapless_navigation_four
        mapless_navigation_four = [
            'xxx/events.out.tfevents.xxx.xxx',
        ]
        data4 = parse_data(mapless_navigation_four, 'MNDS', start_step, end_step, step_interval)

        # model based
        model_based_four = [
            'xxx/events.out.tfevents.xxx.xxx',
        ]
        data5 = parse_data(model_based_four, 'Ours', start_step, end_step, step_interval)

        sns.lineplot(x='step', y='average reward', hue='Methods', data=pd.concat([data5,data2,data1,data3,data4]))

        plt.savefig('result.svg')
    else:
        # parameters
        start_step = 3000
        end_step = 7000
        step_interval = 20

        # model_free_20
        model_free_20 = [
            'xxx/events.out.tfevents.xxx.xxx',
        ]
        data1 = parse_data(model_free_20, 'Ours_w/o_20', start_step, end_step, step_interval)

        # model_free
        model_free = [
            'xxx/events.out.tfevents.xxx.xxx',
        ]
        data2 = parse_data(model_free, 'Ours_w/o_1', start_step, end_step, step_interval)
        
        # fully_distributed
        fully_distributed = [
            'xxx/events.out.tfevents.xxx.xxx',
        ]
        data3 = parse_data(fully_distributed, 'FDMCA', start_step, end_step, step_interval)

        # mapless_navigation
        mapless_navigation = [
            'xxx/events.out.tfevents.xxx.xxx',
        ]
        data4 = parse_data(mapless_navigation, 'MNDS', start_step, end_step, step_interval)

        # model based
        model_based = [
            'xxx/events.out.tfevents.xxx.xxx',
        ]
        data5 = parse_data(model_based, 'Ours', start_step, end_step, step_interval)

        sns.lineplot(x='step', y='average reward', hue='Methods', data=pd.concat([data5,data2,data1,data3,data4]))

        plt.savefig('result-single.svg')