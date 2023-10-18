from LLM_interaction import chat_model
from record import record
import gym
import cv2 as cv
import numpy as np
from typing import List, Union, Optional
import json
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="F:/huggingface_model/qwen/Qwen-14B-Chat-Int4")
parser.add_argument("--epoch", type = str, default="hold")
parser.add_argument("--episode_num", type = int, default=2)
parser.add_argument("--episode_length", type = int, default=200)
parser.add_argument("--render", type=bool, default=True)


class game_agent():
    def __init__(self, modelPath:str, agent_prompt:str) -> None:
        self.prompt = agent_prompt
        self.chater = chat_model(modelPath, cache_dir = "F:/huggingface_model/")

        self.gymA = gym.make("CartPole-v1", render_mode="rgb_array")

    def set_prompt(self, prompt:str):
        self.prompt = prompt

    def interaction(self, state:List[str]) -> str:
        state_str = self.prompt.format(*state)
        action = self.chater.chat(state_str, ",")
        action = action[:-1]
        return action
    
    def step(self, hidden_state = None):
        if hidden_state is None:
            action = self.gymA.action_space.sample()
        else:
            action = self.interaction(hidden_state)
        print("action: {}".format(action))
        action = filter_result(action)
        if action not in self.gymA.action_space:
            return None, None, None, None   
        if action is None:              
            return None, None, None, None                      
        state, reward, done, truncated, info = self.gymA.step(action)
        
        return state, reward, done, action
    
    def reset(self):
        return self.gymA.reset()[0]
    
    def render(self):
        img = self.gymA.render()
        cv.imshow("agent", img)
        cv.waitKey(1)
    
class agent_seq_learning():
    init_prompt = "{}"
    seq_prompt = """{}{}"""
    def __init__(self, modelPath:str, 
                 epoch:Union[int, str], 
                 learning_seq_num:int, 
                 seq_length:int, 
                 render:bool = False) -> None:
        self.epoch = epoch
        self.learning_seq_num = learning_seq_num
        self.seq_length = seq_length
        self.render = render
        
        self.agent = game_agent(modelPath, self.init_prompt)

        self.seq_pool = []

        self.log_record = record("./log.json")
        
    def norm_state(self, state):
        p = (np.clip(state[0], -0.25, 0.25)) / 0.25 * 100 
        v = (np.clip(state[1], -3, 3)) / 3 * 100
        return int(p), int(v)
    
    def get_seqfortrain(self, history = None) -> None:
        agent_seq = []
        reward_seq = []
        for seqIdx in range(self.learning_seq_num):
            state = self.agent.reset()
            episode_reward = 0
            episode_agent = []
            print("new round")
            episode_seq = ""
            for action_idx in range(self.seq_length):
                state_norm = self.norm_state([state[2], state[3]])
                episode_seq += str(state_norm)+": "
                hidden_state = [history, episode_seq]
                print(f"idx: {action_idx}, state: {hidden_state}")                                       
                state, reward, done, action = self.agent.step(hidden_state)
                if state is None:              
                    break   
                episode_reward += reward
                episode_seq += f"{action}, "
                episode_agent.append({"state": state_norm, "action": f"{action}"})
                if self.render:
                    self.agent.render()
                if done:
                    break
                time.sleep(1)
            reward_seq.append(episode_reward)
            agent_seq.append(episode_agent)
        return agent_seq, reward_seq


    def init_system(self):
        agent_seq = []
        reward_seq = []
        for seqIdx in range(self.learning_seq_num):
            state = self.agent.reset()
            episode_reward = 0
            episode_agent = []
            for action_idx in range(self.seq_length):
                state_norm = self.norm_state([state[2], state[3]])
                state, reward, done, action = self.agent.step()
                if state is None:              
                    break   
                episode_reward += reward
                
                episode_agent.append({"state": state_norm, "action": f"{action}"})
                if self.render:
                    self.agent.render()
                if done:
                    break
            reward_seq.append(episode_reward)
            agent_seq.append(episode_agent)
        return agent_seq, reward_seq

    def train(self) -> None:
        learning_seq, reward = self.init_system()           #获取初始的训练序列
        self.update_context(learning_seq, reward)
        current_epoch = 0
        self.agent.set_prompt(self.seq_prompt)
        
        while True:
            if isinstance(self.epoch, int):
                if current_epoch >= self.epoch:
                    break
            elif isinstance(self.epoch, str):
                if self.epoch == "hold":
                    pass
                else:
                    raise ValueError("The epoch needs to be an integer or a string, and can only be a hold when it is a string.")
            learning_seq, reward = self.get_seqfortrain(self.learning_history_format())
            self.log_record.set_record("reward_max", np.max(reward))
            self.log_record.set_record("reward_min", np.min(reward))
            self.log_record.step()
            print("============================================")
            print("reward: {}   last_reward: {}".format(reward, self.seq_pool[-1][0]))
            print("============================================")
            self.update_context(learning_seq, reward)
            current_epoch += 1
            self.log_record.save()
        self.log_record.show_record(["reward_max", "reward_min"])

    def update_context(self, seq, reward):
        last_max_reward = self.seq_pool[-1][0] if len(self.seq_pool)>0 else 0
        for S, R in zip(seq, reward):
            if R > last_max_reward:
                self.seq_pool.append([R, S])
            elif np.random.rand() > 0.5:
                self.seq_pool.append([R, S])

        self.seq_pool = sorted(self.seq_pool, key=lambda x: x[0])


    def learning_history_format(self):
        seq_str = ""
        print(len(self.seq_pool))
        print([S[0] for S in self.seq_pool])
        for i in range(self.learning_seq_num):
            if np.random.rand() > 0.8:   # 扰动，方式局部最优
                i = np.random.choice(len(self.seq_pool), 1)[0]
            for D in self.seq_pool[-i][1]:
                seq_str +=  str(D["state"]) + ": " +D["action"] + ", "

        return seq_str

def filter_result(action):
    return int(action)

def main():
    args = parser.parse_args()
    epoch = int(args.epoch) if args.epoch.isdigit() else args.epoch
    agent_learning = agent_seq_learning(args.model_path, epoch, args.episode_num, args.episode_length, render = args.render)
    agent_learning.train()


if __name__ == "__main__":
    main()
