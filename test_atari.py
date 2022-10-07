import psutil
import argparse
import time, datetime
from tqdm import tqdm, trange

import numpy as np
# import gym
import atari_py
import wandb

from env import Env


def main(env, num, seed, device, wb, args):

    alg_name = 'NL' # configurations['algorithm']['name']
    env_name = env #configurations['environment']['name']
    num_envs = num
    env_domain = 'Atari' # configurations['environment']['domain']

    # group_name = f"RV-Mac-{env_domain}-{env_name}-X{num_envs}"
    # group_name = f"RV-V-{env_domain}-{env_name}-X{num_envs}"
    group_name = f"RV-Q-{env_domain}-{env_name}-X{num_envs}"
    exp_prefix = f"{group_name}-{alg_name}-seed:{seed}"

    print('=' * 50)
    print(f'Start of an RL experiment')
    print(f"\t Environment: {env_name} X {num}")
    print(f"\t Random seed: {seed}")
    print('=' * 50)

    if wb:
        wandb.init(
            group=group_name,
            name=exp_prefix,
            project=f'RL-WC'
        )

    LS = int(1e6)

    env = Env(args)
    env.train()
    action_space = env.action_space()

    # envs = gym.vector.make(env_name, num_envs=num)
    o, terminated = env.reset(), False
    env_steps = 0
    LT = trange(1, LS+1, desc=env_name, position=0)
    # SPS = tqdm(desc='SPS', position=1, colour='PURPLE')
    CPU = tqdm(total=100, desc='CPU %', position=1, colour='RED')
    RAM = tqdm(total=100, desc='RAM %', position=2, colour='BLUE')
    CPUList, RAMList, SPSList = [], [], []
    logs = dict()

    if num == 0: num = 1

    start_time_real = time.time()

    with CPU, RAM:
        for t in LT:
            if terminated: o = env.reset()
            # a = env.action_space.sample()
            a = np.random.randint(0, action_space)
            o_next, r, terminated = env.step(a)
            env_steps += num
            if t%500==0:
                cur_time_real = time.time()
                total_time_real = cur_time_real - start_time_real
                sps = env_steps//total_time_real
                # SPS.n = sps
                # SPS.refresh()
                SPSList.append(sps)
                # print(f'env_steps={env_steps} | time={total_time_real} | sps={sps}')

                CPU.n = psutil.cpu_percent()
                CPU.refresh()
                RAM.n = psutil.virtual_memory().percent
                RAM.refresh()
                CPUList.append(CPU.n)

                logs['hardware/cpu                        '] = CPU.n
                logs['hardware/cpu-avg                    '] = np.mean(CPUList)
                logs['hardware/ram                        '] = RAM.n
                logs['time/total                          '] = total_time_real
                logs['time/sps                            '] = sps
                logs['time/sps-avg                        '] = np.mean(SPSList)
                LT.set_postfix({'Steps': env_steps,'SPS': sps})
                if wb: wandb.log(logs, step=env_steps)
            if env_steps >= LS: break
    env.close()


    print('\n')
    print('End of the RL experiment')
    print('=' * 50)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    # parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument('--id', type=str, default='default', help='Experiment ID')

    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--game', type=str, default='pong', choices=atari_py.list_games(), help='ATARI game')
    parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
    parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')

    parser.add_argument('--render', action='store_true', help='Display screen (testing only)')

    parser.add_argument('-env', type=str, default="pong")
    parser.add_argument('-num', type=int, default=0)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-device', type=str, default='cpu')
    parser.add_argument('-wb', action='store_true')

    args = parser.parse_args()

    env = args.env
    num = args.num
    seed = args.seed
    device = args.device
    wb = args.wb

    main(env, num, seed, device, wb, args)
