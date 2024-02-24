# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:48:53 2024

@author: jaege
"""

import numpy as np

class SimpleGridEnvironment:
    def __init__(self):
        self.grid_size = (5, 5)
        self.start_state = (0, 0)
        self.goal_state = (4, 4)
        self.current_state = self.start_state
        self.obstacle_states = [(4, 2), (2, 2), (1, 3)]

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        new_state = self.current_state

        # 이동 방향: 0(상), 1(우), 2(하), 3(좌)
        if action == 0 and self.current_state[0] > 0:
            new_state = (self.current_state[0] - 1, self.current_state[1])
        elif action == 1 and self.current_state[1] < self.grid_size[1] - 1:
            new_state = (self.current_state[0], self.current_state[1] + 1)
        elif action == 2 and self.current_state[0] < self.grid_size[0] - 1:
            new_state = (self.current_state[0] + 1, self.current_state[1])
        elif action == 3 and self.current_state[1] > 0:
            new_state = (self.current_state[0], self.current_state[1] - 1)

        # 보상은 목표 도달 시 1, 그 외에는 0
        reward = 1 if new_state == self.goal_state else 0

        # 장애물에 부딪힌 경우 음의 보상
        if new_state in self.obstacle_states:
            reward = -1

        # 에피소드 종료 여부는 목표 도달 시
        done = new_state == self.goal_state

        # 상태 업데이트
        self.current_state = new_state

        return self.current_state, reward, done
class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_prob=1.0, exploration_decay=0.995, exploration_min=0.01):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min

        # Q 테이블 초기화
        self.q_table = np.zeros((num_states[0], num_states[1], num_actions))

    def choose_action(self, state):
        # Epsilon-greedy exploration
        if np.random.rand() < self.exploration_prob:
            return np.random.choice(self.num_actions)
        else:
            # Randomly choose an action if multiple actions have the same Q-value
            max_q_value = np.max(self.q_table[state[0], state[1], :])
            best_actions = np.where(self.q_table[state[0], state[1], :] == max_q_value)[0]
            return np.random.choice(best_actions)
        
    def update_q_table(self, state, action, reward, next_state):
        # Q-learning 업데이트 규칙 적용
        current_q = self.q_table[state[0], state[1], action]
        best_next_q = np.max(self.q_table[next_state[0], next_state[1], :])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * best_next_q)
        self.q_table[state[0], state[1], action] = new_q

        # 탐험 확률 감소
        if self.exploration_prob > self.exploration_min:
            self.exploration_prob *= self.exploration_decay

# 미로 환경과 에이전트 초기화
env = SimpleGridEnvironment()
num_states = env.grid_size
num_actions = 4  # 이동 방향: 0(상), 1(우), 2(하), 3(좌)

agent = QLearningAgent(num_states, num_actions, exploration_prob=0.5)
path = []

# 학습
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        # 에이전트가 현재 상태에서 행동 선택
        action = agent.choose_action(state)

        # 선택한 행동을 환경에 적용하고 새로운 상태, 보상, 종료 여부를 얻음
        next_state, reward, done = env.step(action)

        # 경로에 현재 상태 추가
        path.append(state)

        # Q 테이블 업데이트
        agent.update_q_table(state, action, reward, next_state)

        total_reward += reward
        state = next_state

        if done:
            break

    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")

# 학습된 정책을 이용하여 환경에서 에이전트 테스트
test_episodes = 5
for episode in range(test_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        total_reward += reward
        state = next_state

        if done:
            break

    print(f"Test Episode: {episode}, Total Reward: {total_reward}")
    
# Print the final Q-table
print("\nFinal Q-table:")
print(agent.q_table[4,:])

import matplotlib.pyplot as plt
import numpy as np

# Assuming obstacle_states and goal_state are defined somewhere in your script
obstacle_states = [(4, 2), (2, 2), (1, 3)]
goal_state = (4, 4)

# Grid 생성
grid_size = (5, 5)
grid = np.zeros(grid_size)

# 장애물 표시
for obs in obstacle_states:
    grid[obs[0], obs[1]] = -1

# 목표 지점 표시
grid[goal_state[0], goal_state[1]] = 2

# 최적 경로 찾기
state = env.reset()
optimal_path = [state]

while state != goal_state:
    action = np.argmax(agent.q_table[state[0], state[1], :])
    next_state, _, _ = env.step(action)
    optimal_path.append(next_state)
    state = next_state

# 에이전트 경로 표시
for step in optimal_path:
    grid[step[0], step[1]] = 1

# 그리기
plt.figure(figsize=(4, 4))
plt.title("Optimal Path Based on Final Q-Table")
plt.imshow(grid, cmap='viridis', origin='lower', interpolation='none')

# 각 셀에 숫자로 값 표시
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        plt.text(j, i, grid[i, j], ha='center', va='center', color='white' if grid[i, j] != -1 else 'black')

plt.show()