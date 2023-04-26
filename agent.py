import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        # 这是一个贪吃蛇游戏中获取当前状态的函数，
        # 将当前贪吃蛇头部四周以及前方的空间状态和食物相对于贪吃蛇头部的位置状态，作为状态向量，用来作为强化学习的输入。
        # 具体来说，该函数通过判断当前贪吃蛇头部四周的空间状态是否为危险（即是否会撞到障碍物），
        # 以及当前贪吃蛇前进方向和当前贪吃蛇头部四周的空间状态是否有障碍物来得出三个状态。
        # 接着，该函数用一个4维向量表示当前贪吃蛇的运动方向，一个4维向量表示食物在相对于贪吃蛇头部的位置，
        # 将这些信息合并为一个长度为11的向量返回。
        head = game.snake[0] # 贪吃蛇头部
        point_l = Point(head.x - 20, head.y) # 贪吃蛇头部左边的点（20）
        point_r = Point(head.x + 20, head.y) # 贪吃蛇头部右边的点（20）
        point_u = Point(head.x, head.y - 20) # 贪吃蛇头部上边的点（20）
        point_d = Point(head.x, head.y + 20) # 贪吃蛇头部下边的点（20）
        
        dir_l = game.direction == Direction.LEFT  # 这行代码是在检查当前蛇的移动方向是否是左边，返回的是一个bool类型的值。如果蛇的移动方向是左边，dir_l的值就是True，否则为False。
        dir_r = game.direction == Direction.RIGHT # 
        dir_u = game.direction == Direction.UP    
        dir_d = game.direction == Direction.DOWN  

        state = [
            # Danger straight，前方危险
            (dir_r and game.is_collision(point_r)) or # 如果蛇的移动方向是右边，且右边有障碍物，那么dir_r的值就是True，否则为False。如果dir_r的值是True，那么game.is_collision(point_r)的值就是True，否则为False。如果dir_r和game.is_collision(point_r)的值都是True，那么(dir_r and game.is_collision(point_r))的值就是True，否则为False。
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right，
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction 移动方向
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done): #储存状态，动作，奖励，下一个状态，是否结束
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    # 迭代一次，训练一次
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state): # 行为选择函数
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state，这里的state是一个list，包含了11个元素，分别是
        state_old = agent.get_state(game)  # game返回的是一个list，包含了3个元素，分别是蛇的位置，食物的位置，蛇的移动方向

        # get move, 根据state_old，选择一个动作
        final_move = agent.get_action(state_old)

        # perform move and get new state, 环境接受动作，返回新的状态，奖励，是否结束，得分
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory，训练短期记忆，走一步训练一步
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember，将状态，动作，奖励，下一个状态，是否结束存储到memory中
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory() # 长期记忆，走完一局训练一次

            if score > record: # 记录最高分
                record = score
                agent.model.save() # 保存模型

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores) # 画图,每一局的得分和平均得分


if __name__ == '__main__':
    train()