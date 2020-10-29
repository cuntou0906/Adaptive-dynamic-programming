import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

state_dim = 2                                                                  # 状态维度
v_dim = 1                                                                      # 价值维度
action_dim = 1                                                                 # 动作维度
learing_rate = 0.005                                                           # 学习率
# learing_num = 2
learing_num = 200                                                              # 学习次数
sim_num = 20                                                                   # 仿真步长
x0 = np.array([2,-1])                                                          # 初始状态
epislon = 1.4                                                                  # 阈值

torch.manual_seed(1)                                                           #设置随机种子，使得每次生成的随机数是确定的

########################################################################################################################
# 定义神经网络类
########################################################################################################################
class Model(torch.nn.Module):
    # 初始化
    def __init__(self):
        super(Model, self).__init__()
        self.lay1 = torch.nn.Linear(state_dim, 10, bias = False)               # 线性层
        self.lay1.weight.data.normal_(0,0.5)                                   # 权重初始化
        self.lay2 = torch.nn.Linear(10, 1, bias = False)                       # 线性层
        self.lay2.weight.data.normal_(0, 0.5)                                  # 权重初始化

    def forward(self, x):
        layer1 = self.lay1(x)                                                  # 第一隐层
        layer1 = torch.nn.functional.relu(layer1)                              # relu激活函数
        output = self.lay2(layer1)                                             #输出层
        return output

########################################################################################################################
# 定义价值迭代类
########################################################################################################################
class Value_Iteration():
    def __init__(self):
        self.V_model = Model()                                                  # 定义V网络
        self.A_model = Model()                                                  # 定义A网络
        self.criterion = torch.nn.MSELoss(reduction='mean')                     # 平方误差损失
        self.optimizer1 = torch.optim.SGD(self.A_model.parameters(), lr=learing_rate)    # 利用梯度下降算法优化model.parameters
        self.optimizer2 = torch.optim.SGD(self.V_model.parameters(), lr=learing_rate)    # 利用梯度下降算法优化model.parameters

        # 采样 状态点
        x = np.arange(-2, 2, 0.1)
        y = np.arange(-1, 1, 0.1)
        xx, yy = np.meshgrid(x, y)                                              # 为一维的矩阵
        self.state = np.transpose(np.array([xx.ravel(), yy.ravel()]))           # 所有状态
        self.state_num = self.state.shape[0]                                    # 状态个数
        self.cost = []                                                          # 初始化误差矩阵
        pass

    ####################################################################################################################
    # 定义模型函数
    ####################################################################################################################
    def model(self, current_state, u):
        next_state = np.zeros([current_state.shape[0], current_state.shape[1]])  # 初始化下一个状态
        for index in range(current_state.shape[0]):                              # 对每个样本计算下一个状态 根据输入的u
            next_state[index, 0] = 0.2 * current_state[index, 0] * np.exp(current_state[index, 1] ** 2)
            next_state[index, 1] = 0.3 * current_state[index, 1] ** 3 - 0.2 * u[index]
            pass
        return next_state

    ####################################################################################################################
    # 定义学习函数
    ####################################################################################################################
    def learning(self):
       for index in range(learing_num):
           print('the ',index,' --th  learing start')

           last_V_value = self.V_model(Variable(torch.Tensor(self.state))).data

           #############################################################################################################
           # 更新Actor网络
           #############################################################################################################
           A_predict = self.A_model( Variable(torch.Tensor(self.state)))          # 预测值
           AA_predict = A_predict * A_predict

           la_u = self.A_model(Variable(torch.Tensor(self.state)))
           la_next_state = self.model(self.state, la_u)                           # 计算下一时刻状态
           AA_target = np.zeros([self.state_num, 1])                              # 初始化A网络的标签
           for index in range(self.state_num):                                    # 循环计算所有状态的标签
               next_V = self.V_model(Variable(torch.Tensor(la_next_state[index, :])))
               AA_target[index] = self.state[index, 0] ** 2 + self.state[index, 1] ** 2 + next_V.data
               pass
           # print(la_V_label)
           # print(AA_target.shape)
           # print(AA_predict.size())
           loss1 = self.criterion(AA_predict, Variable(torch.Tensor(AA_target)))   # 计算损失
           self.optimizer1.zero_grad()                                             # 对模型参数做一个优化，并且将梯度清0
           loss1.backward()                                                        # 计算梯度
           self.optimizer1.step()                                                  # 权重更新

           #############################################################################################################
           # 更新Crictic网络
           #############################################################################################################
           V_predict = self.V_model(Variable(torch.Tensor(self.state)))            # 预测值

           la_u = self.A_model(Variable(torch.Tensor(self.state)))                 # 计算输入
           la_next_state = self.model(self.state, la_u)                            # 计算下一时刻状态
           V_target = np.zeros([self.state_num, 1])                                # 初始化V网络的标签
           for index in range(self.state_num):                                     # 循环计算所有状态的标签
               next_V = self.V_model(Variable(torch.Tensor(la_next_state[index, :])))
               V_target[index] = self.state[index, 0] ** 2 + self.state[index, 1] ** 2 + la_u.data[index] ** 2 + next_V.data
               pass
           # print(la_V_label)

           loss2 = self.criterion(V_predict, Variable(torch.Tensor(V_target)))      # 计算损失
           self.optimizer2.zero_grad()                                              # 对模型参数做一个优化，并且将梯度清0
           loss2.backward()                                                         # 计算梯度
           self.optimizer2.step()                                                   # 权重更新

           V_value = self.V_model(Variable(torch.Tensor(self.state))).data          # 计算V
           pp = np.abs(V_value)-np.abs(last_V_value)
           #print(pp)
           dis = np.sum(np.array(pp.reshape(self.state_num)))                       #平方差
           self.cost.append(dis)
           print('平方差', dis)
           if dis < epislon:
               break
           pass
       pass

    #######################################################################################################
    # 定义仿真函数
    # 通过得到的Actor选择动作
    # 同时利用Critic计算V
    #######################################################################################################
    def simulator(self):
        print('the simulation is start')
        #self.restore(self.path)
        State_traject = np.zeros([sim_num + 1, state_dim])
        State_traject[0, :] = x0
        u_traject = np.zeros([sim_num, 1])
        for index in range(sim_num):
            print('the ', index, ' --th  time start')
            # print(State_traject[index,:])
            print('当前状态：', Variable(torch.Tensor(State_traject[index,:])).data)
            sim_actor = self.A_model(Variable(torch.Tensor(State_traject[index,:])))
            print('当前输入：',sim_actor)
            u_traject[index] = sim_actor.data
            # print(State_traject[index,:])
            sim_nexstate = self.model(State_traject[index, :].reshape(1, 2), sim_actor.data)
            print('下一时刻状态：', sim_nexstate)
            State_traject[index + 1, :] = sim_nexstate
            pass
        pass
        V_traject = self.V_model(Variable(torch.Tensor(State_traject))).data
        print('the simulation is over')
        self.plot_curve(State_traject, u_traject, V_traject,self.cost)
        pass

    #######################################################################################################
    # 绘图函数
    # 分别绘制状态轨迹 控制输入u轨迹 值函数V轨迹
    # 并将结果保存！
    #######################################################################################################
    def plot_curve(self, s, u, V,cost):
        # print('\nstate\n',s)
        # print('\nu\n', u)
        # print('\nV\n', V)
        # 绘制状态轨迹
        plt.figure(1)
        plt.plot(s[:, 0], 'r', label='State_1')
        plt.plot(s[:, 1], 'b', label='State_2')
        plt.title('State_Trajecteory')
        plt.xlabel('iter')
        plt.ylabel('State')
        plt.legend()
        plt.grid()
        plt.savefig(r'ADPresultfig\VI_state.png')
        plt.show()

        # 绘制控制输入u轨迹
        plt.figure(2)
        plt.plot(u, )
        plt.title('U_Trajecteory')
        plt.xlabel('iter')
        plt.ylabel('u')
        plt.grid()
        plt.savefig(r'ADPresultfig\VI_u.png')
        plt.show()

        # 绘制值函数V的轨迹
        plt.figure(3)
        plt.plot(V, 'r')
        plt.title('Cost_Trajecteory')
        plt.xlabel('iter')
        plt.ylabel('Cost')
        plt.grid()
        plt.savefig(r'ADPresultfig\VI_V.png')
        plt.show()

        # 绘制值函数V的轨迹
        plt.figure(4)
        plt.plot(cost, 'r')
        plt.title('Train_loss_Trajecteory')
        plt.xlabel('iter')
        plt.ylabel('Train_loss')
        plt.grid()
        plt.savefig(r'ADPresultfig\VI_loss.png')
        plt.show()
        pass

########################################################################################################################
# 函数起始运行
# 在仿真时候，直接调用最优的模型进行仿真
# 最优的模型根据损失函数进行判断
########################################################################################################################
if __name__ == '__main__':
    Agent = Value_Iteration()                                                # 值迭代类实例化
    Agent.learning()                                                         # 学习
    Agent.simulator()                                                        # 仿真
