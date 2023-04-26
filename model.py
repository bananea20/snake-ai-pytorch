import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module): 
    # Qnet通常指的是“Q网络”（Q-network），
    # 是一种基于神经网络的强化学习算法。
    # 它可以用来解决基于状态-行动值函数（state-action value function）的强化学习问题，
    # 其中策略和价值函数通过学习和迭代来优化。

    # Q网络的主要思想是使用神经网络来逼近状态-行动值函数，即Q值。
    # 通过在神经网络中输入状态，输出每个可能行动的Q值，
    # 然后选择最大Q值对应的行动来更新策略和价值函数。
    # Q网络通常与深度学习结合使用，
    # 就是根据当前状态来预测下一步的行动

    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer: # Q学习器
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state) # 通过Q网络得到预测的各个动作的价值

        target = pred.clone() # 克隆预测动作的价值
        
        for idx in range(len(done)): # len(done) 表示游戏进行了多少步
            Q_new = reward[idx]      # 每一步的得到的奖励为Q_new
            if not done[idx]:        # 如果游戏没有结束
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                # Q_new = r + y * max(next_predicted Q value)
                # Q_new = 该步的奖励 + 折扣因子 * 下一步的状态的各个动作的最大价值
                # 使用当前状态和动作获得的奖励作为监督信号
                # 当前动作获得的奖励的表达方式为 torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            # 对于每一步，将当前动作的价值更新为Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        # 计算target和pred之间的损失值，得到loss
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()



