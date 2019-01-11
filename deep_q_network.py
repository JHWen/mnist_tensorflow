import random
import time
from collections import deque
from keras.layers import *
from keras.models import Model
import keras
import numpy as np
from tqdm import tqdm
from keras.utils import np_utils
from sklearn.metrics import accuracy_score


# 环境
class MnEnviroment(object):
    def __init__(self, x, y):
        self.train_X = x
        self.train_Y = y
        self.current_index = self._sample_index()
        self.action_space = len(set(y)) - 1

    def reset(self):
        obs, _ = self.step(-1)
        return obs

    '''
    action: 0-9 categori, -1 : start and no reward
    return: next_state(image), reward
    '''

    def step(self, action):
        if action == -1:
            _c_index = self.current_index
            self.current_index = self._sample_index()
            return self.train_X[_c_index], 0
        r = self.reward(action)
        self.current_index = self._sample_index()
        return self.train_X[self.current_index], r

    def reward(self, action):
        c = self.train_Y[self.current_index]
        # print(c)
        return 1 if c == action else -1

    def sample_actions(self):
        return random.randint(0, self.action_space)

    def _sample_index(self):
        return random.randint(0, len(self.train_Y) - 1)


class DQN(object):
    def __init__(self):
        self.epoches = 2000
        self.replay_size = 64
        self.pre_train_num = 256
        self.gamma = 0.
        self.alpha = 0.5
        self.forward = 512
        self.epislon_total = 2018
        self.memory = deque(maxlen=512)
        self.every_copy_step = 128
        self.dummy_actions = None
        self.num_actions = None

        self.image_width = 28
        self.image_hidth = 28
        self.num_actions = 10

        self.actor_q_model = None
        self.actor_model = None

    def createDQN(self, input_width, input_height, actions_num):
        img_input = Input(shape=(input_width, input_height, 1), dtype='float32', name='image_inputs')
        # conv1
        conv1 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(img_input)
        conv2 = Conv2D(64, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv1)
        conv3 = Conv2D(64, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv2)
        conv4 = Conv2D(128, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv3)
        x = Flatten()(conv4)
        x = Dense(128, activation='relu')(x)
        outputs_q = Dense(actions_num, name='q_outputs')(x)
        # one hot input
        actions_input = Input((actions_num,), name='actions_input')
        q_value = multiply([actions_input, outputs_q])
        q_value = Lambda(lambda l: K.sum(l, axis=1, keepdims=True), name='q_value')(q_value)

        model = Model(inputs=[img_input, actions_input], outputs=q_value)
        model.compile(loss='mse', optimizer='adam')
        return model

    def remember(self, state, action, action_q, reward, next_state):
        self.memory.append([state, action, action_q, reward, next_state])

    def pre_remember(self, env, pre_go=30):
        state = env.reset()
        for i in range(pre_go):
            rd_action = env.sample_actions()
            next_state, reward = env.step(rd_action)
            self.remember(state, rd_action, 0, reward, next_state)
            state = next_state

    def epsilon_calc(self, step, ep_min=0.01, ep_max=1, esp_total=1000):
        return max(ep_min, ep_max - (ep_max - ep_min) * step / esp_total)

    def epsilon_greedy(self, env, state, step, ep_min=0.01, ep_total=1000):
        epsilon = self.epsilon_calc(step, ep_min, 1, ep_total)
        if np.random.rand() < epsilon:
            return env.sample_actions(), 0
        qvalues = self.get_q_values(self.actor_q_model, state)
        return np.argmax(qvalues), np.max(qvalues)

    def get_q_values(self, model_, state):
        inputs_ = [state.reshape(1, *state.shape), self.dummy_actions]
        qvalues = model_.predict(inputs_)
        return qvalues[0]

    def sample_ram(self, sample_num):
        return np.array(random.sample(self.memory, sample_num))

    def replay(self):
        if len(self.memory) < self.replay_size:
            return
        # 从记忆中i.i.d采样
        samples = self.sample_ram(self.replay_size)
        # 展开所有样本的相关数据
        # 这里next_states没用 因为和上一个state无关。
        states, actions, old_q, rewards, next_states = zip(*samples)
        states, actions, old_q, rewards = np.array(states), np.array(actions).reshape(-1, 1), \
                                          np.array(old_q).reshape(-1, 1), np.array(rewards).reshape(-1, 1)

        actions_one_hot = np_utils.to_categorical(actions, self.num_actions)
        # print(states.shape,actions.shape,old_q.shape,rewards.shape,actions_one_hot.shape)
        # 从actor获取下一个状态的q估计值 这里也没用 因为gamma=0 也就是不对bellman方程展开
        # inputs_ = [next_states,np.ones((replay_size,num_actions))]
        # qvalues = actor_q_model.predict(inputs_)

        # q = np.max(qvalues,axis=1,keepdims=True)
        q = 0
        q_estimate = (1 - self.alpha) * old_q + self.alpha * (rewards.reshape(-1, 1) + self.gamma * q)
        history = self.actor_model.fit([states, actions_one_hot], q_estimate, epochs=1, verbose=0)
        return np.mean(history.history['loss'])

    def build_model(self):
        # init mode
        self.actor_model = self.createDQN(self.image_width, self.image_hidth, self.num_actions)  # 用于决策
        self.actor_q_model = Model(inputs=self.actor_model.input,
                                   outputs=self.actor_model.get_layer('q_outputs').output)

    def train(self, mnist):
        x_train, y_train = mnist['x_train'], mnist['y_train']

        # reshape num*28*28*1
        x_train = x_train.reshape(*x_train.shape, 1)

        # normalization
        x_train = x_train / 255.

        self.dummy_actions = np.ones((1, self.num_actions))

        # start environment
        env = MnEnviroment(x_train, y_train)

        self.memory.clear()
        reward_rec = []
        self.pre_remember(env, self.pre_train_num)

        bar = tqdm(range(1, self.epoches + 1))
        state = env.reset()

        for epoch in bar:
            total_rewards = 0
            epo_start = time.time()
            for step in range(self.forward):
                # 对每个状态使用epsilon_greedy选择
                action, q = self.epsilon_greedy(env, state, epoch, ep_min=0.01, ep_total=self.epislon_total)
                eps = self.epsilon_calc(epoch, esp_total=self.epislon_total)
                # play
                next_state, reward = env.step(action)
                # 加入到经验记忆中
                self.remember(state, action, q, reward, next_state)
                # 从记忆中采样回放，保证iid。实际上这个任务中这一步不是必须的。
                loss = self.replay()
                total_rewards += reward
                state = next_state
            reward_rec.append(total_rewards)
            bar.set_description(
                'R:{} L:{:.4f} T:{} P:{:.3f}'.format(total_rewards, loss, int(time.time() - epo_start), eps))

        # save model
        self.actor_model.save('./model_path/critic_2000.HDF5')

        # test
        x_test, y_test = mnist['x_test'], mnist['y_test']
        y_pred = self.predict(self.actor_q_model, x_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print('mnist test accuracy: %f' % test_accuracy)

    def predict(self, model, states):
        inputs_ = [states, np.ones(shape=(len(states), self.num_actions))]
        qvalues = model.predict(inputs_)
        return np.argmax(qvalues, axis=1)

    def test(self, mnist):
        self.actor_model = keras.models.load_model('./model_path/critic_2000.HDF5')
        self.actor_q_model = Model(inputs=self.actor_model.input,
                                   outputs=self.actor_model.get_layer('q_outputs').output)
        x_test, y_test = mnist['x_test'], mnist['y_test']
        x_test = x_test.reshape(*x_test.shape, 1)
        x_test = x_test / 255.

        y_pred = self.predict(self.actor_q_model, x_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print('mnist test accuracy: %f' % test_accuracy)


if __name__ == '__main__':
    # load dataset
    mnist_dataset = np.load('./data/mnist.npz')

    dqn = DQN()
    dqn.build_model()
    # dqn.train(mnist_dataset)
    dqn.test(mnist_dataset)
