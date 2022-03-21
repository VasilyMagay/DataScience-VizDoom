# Импортируем все функции из cреды ViZDoom
from vizdoom import *

# Библиотека TensorFlow
import tensorflow as tf

# Библиотека NumPy
import numpy as np

# Импортируем модуль для генерации рандомных значений
import random

# Модуль для сохранения результатов в файл
import pickle

# Модуль для работы с картинками и видео (нужен для предобработки данных и записи результата)
import cv2

# Тип данных deque (список, где автоматически удаляются старые значения при добавлении новых,
# чтобы не переполнять память)
from collections import deque

# Модуль pyplot из библиотеки Matplotlib
# import matplotlib.pyplot as plt

# Функция для создания и загрузки модели из TensorFlow
from tensorflow.keras.models import load_model, Model, Sequential

# Все слои из Keras
from tensorflow.keras.layers import *

# Функция для удобного onehot энкодинга
from tensorflow.keras.utils import to_categorical

# Функции для воспроизведения видео с результатом
from moviepy.editor import *

CONTENT_PATH = os.path.join(os.path.abspath(os.curdir), 'content')
LIB_PATH = os.path.join(CONTENT_PATH, 'ViZDoom')

MODEL_FILE = os.path.join(CONTENT_PATH, 'vizdoom_DQN_model.h5')
STATS_FILE = os.path.join(CONTENT_PATH, 'vizdoom_DQN_stats.txt')


class ViZDoomModel:

    def __init__(self):

        # Гиперпараметры для обучения нашей DQN-нейросети

        self.gamma = 0.95  # Гамма
        self.observation_steps = 10000  # Количество шагов
        self.target_model_update_frequency = 5000  # Частота обновления целевой

        # Другие гиперпараметры

        initial_epsilon = 1  # Начальное значение эпсилона (вероятность принять рандомный шаг)
        self.epsilon = initial_epsilon  # Текущее значение эпсилона (инициализируется как стартовое значение)
        self.final_epsilon = 0.01  # Минимальное значение эпсилона (должен быть выше 0)
        epsilon_decay_steps = 200000  # Мы будем уменьшать значение эпсилона на протяжении 200 000 шагов
        # Задаем количество, на которое будем снижать эпсилон после каждого шага
        self.epsilon_decay_factor = (initial_epsilon - self.final_epsilon) / epsilon_decay_steps

        self.timesteps_per_train = 100  # Обучаем модель раз в 100 шагов (не обязательно ждать до конца игры)
        self.learning_rate = 1e-4  # Обычно в обучении с подкреплением ставят низкий шаг обучения, например 1e-4
        self.batch_size = 32  # Используем размер пакета в 32

        # Параметры для преобразования картинок с кадрами

        self.image_width = 64  # Ширина картинки (кадра)
        self.image_height = 64  # Высота картинки (кадра)
        self.num_frames = 4  # Количество последовательных кадров в одном состоянии (используется позже)
        # Размерность каждого состояния — размер картинки
        self.state_shape = (self.image_width, self.image_height, self.num_frames)

        # Запись информации в память

        # В памяти будет храниться не более 40 000 пар текущих и следующих состояний,
        # действия которых нейронная сеть выбрала, а также их соответствующие награды
        maximum_memory_length = 40000

        self.memory = deque([], maxlen=maximum_memory_length)  # Создаем буфер памяти

        # Устанавливаем количество кадров за каждое действие. Нам не нужен каждый кадр,
        # поэтому будем совершать действие и брать новое состояние лишь раз в 4 кадра:
        self.frames_per_action = 4

        # Переменные, используемые позже
        self.main_model = None
        self.target_model = None
        self.game = None

        self.record_rewards = []  # Сюда будем записывать награды за 10 эпизодов (для анализа статистики)
        self.record_kills = []  # Сюда будем записывать количество убитых врагов (для анализа статистики)
        self.record_ammos = []  # Сюда будем записывать количество оставшихся патронов (для анализа статистики)
        self.episode_number = 1  # Инициализируем номер эпизода как 1
        self.timestep = 0  # Инициализируем номер шага как 0

        self.action_size = 0  # количество возможных дискретных действий в среде

    def make_dqn(self):
        """
        Функция создания нейросети, будет использована для создания основной и целевой моделей
        Args:
        Returns: возвращает готовую, компилированную модель
        """
        model = Sequential()

        model.add(Conv2D(32, 8, strides=(4, 4), activation='relu', input_shape=(self.state_shape)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

        model.add(Conv2D(64, 4, strides=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

        model.add(Conv2D(64, 4, activation='relu'))

        model.add(Flatten())

        model.add(Dense(256, activation='relu'))

        # Выходной слой должен иметь активационную функцию 'linear' — мы предсказываем награды на выходе НС.
        model.add(Dense(self.action_size, activation='linear'))

        # Практика показывает, что RMSprop — хороший оптимизатор для обучения с подкреплением,
        # однако можно использовать adam.
        optimizer = tf.keras.optimizers.RMSprop(lr=self.learning_rate)

        # Компилируем модель с функцией ошибки mse и заданным оптимизатором.
        model.compile(loss='mse', optimizer=optimizer)

        return model

    def update_target_model(self):
        """
        Функция обновления весов в целевой модели, т. е. той, что
        устанавливает веса целевой модели (которая не обучается) такими
        же, как веса основной модели (которая обучается)
        """
        self.target_model.set_weights(self.main_model.get_weights())

    def preprocess_frame(self, frame):
        """
        Функция преобразования изображений
        Args:
          frame -
        Returns:
          Возвращаем предобработанное, нормализованное, решейпнутое изображение
        """
        # Меняем оси:
        frame = np.rollaxis(frame, 0, 3)

        # Меняем размерность картинки на (64×64):
        frame = cv2.resize(frame, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)

        # Переводим в черно-белое:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame.reshape(self.image_width, self.image_height, 1) / 255

    @staticmethod
    def get_reward(previous_info, current_info, episode_done):
        """
        Функция предобработки наград
        Args:
            previous_misc — информация об игровой среде на предыдущем кадре (количество убитых врагов, патроны и здоровье)
            current_misc — информация об игровой среде на текущем кадре (количество убитых врагов, патроны и здоровье)
            episode_done — булевое значение, которое говорит, если кадр последний в эпизоде.
            misc[0] — количество убитых врагов, misc[1] — патроны, misc[2] — здоровье
        Returns: подсчитанная награда
        """

        # Инициализируем награду как 0
        reward = 0

        # Если кадр последний в игре, ставим награду как -0.1 и возвращаем ее (агент умер)
        if episode_done:
            reward = -0.1

            return reward

        # Если убили врага в кадре, увеличиваем награду на 1
        if current_info[0] > previous_info[0]:
            reward += 1

        # Если потеряли здоровье, уменьшаем награду на 0.1
        if current_info[1] < previous_info[1]:
            reward -= 0.1

        # Если использовали патрон, уменьшаем награду на 0.1
        if current_info[2] < previous_info[2]:
            reward -= 0.1

        return reward

    def get_action(self, state, epsilon_prob):
        """
        Функция предсказания награды за действие
        Args: state -
              epsilon -
        Returns: выбранное действие и новое значение epsilon
        """

        # Генерируем рандомное значение и сравниваем
        if random.random() <= epsilon_prob:
            action_index = np.random.randint(0, self.action_size)
        else:  # Иначе (если рандомное число больше, чем эпсилон)
            # Предсказываем все Q-значения при следующим действии (Q(s, a) для каждого действия a)
            q_values = self.main_model.predict(np.expand_dims(state, axis=0))
            # Извлекаем индекс действия который приводит к максимальному значению Q(s, a)
            action_index = np.argmax(q_values)

        # Снижаем значение эпсилон, если оно больше, чем final_epsilon,
        # снижаем значение epsilon на epsilon_decay_factor.
        if epsilon_prob > self.final_epsilon:
            epsilon_prob -= self.epsilon_decay_factor

        return action_index, epsilon_prob

    def add_to_memory(self, previous_state, action, reward, current_state, episode_done):
        """
        Функция записи информации в память
        Args:
          previous_state — массивы из состояния среды
          action — действие, которое было в нем принято
          reward — награда, которая была получена
          current_state — следующее состояние, к которому действие привело
          episode_done — булевое значение флагов окончания игры (кадр последний в эпизоде)
        Returns:
        """
        # memory — глобальная переменная. Мы записываем в нее всю нужную информацию:
        self.memory.append((previous_state, action, reward, current_state, episode_done))

    def sample_from_memory(self):
        """
        Функция сэмплирования данных
        Args:
        Returns: распакованные данные
        """

        # Определим размер памяти:
        memory_batch_size = min(self.batch_size * self.timesteps_per_train, len(self.memory))

        # Сэмплируем данные:
        mini_batch = random.sample(self.memory, memory_batch_size)

        # Создаем массив из нулей с размерностью предыдущих состояний, массива действий,
        # массива наград, текущих состояний, флагов окончания игры
        previous_states = np.zeros((memory_batch_size, self.image_width, self.image_height, self.num_frames))
        actions = np.zeros(memory_batch_size)
        rewards = np.zeros(memory_batch_size)
        current_states = np.zeros((memory_batch_size, self.image_width, self.image_height, self.num_frames))
        episode_done = np.zeros(memory_batch_size)

        # Перебираем данные и копируем их значения в массивы нулей:
        for i in range(memory_batch_size):
            previous_states[i, :, :, :] = mini_batch[i][0]
            actions[i] = mini_batch[i][1]
            rewards[i] = mini_batch[i][2]
            current_states[i, :, :, :] = mini_batch[i][3]
            episode_done[i] = mini_batch[i][4]

        return previous_states, actions.astype(np.uint8), rewards, current_states, episode_done

    def train_network(self):
        """
        Функция обучения алгоритма
        Args:
        Returns: обученная модель
        """

        # Извлекаем пакет данных из памяти:
        previous_states, actions, rewards, current_states, game_finished = self.sample_from_memory()

        # Предсказываем Q(s, a):
        q_values = self.main_model.predict(previous_states)

        # Предсказываем Q(s', a'):
        next_q_values = self.target_model.predict(current_states)

        # Модифицируем значения Q:
        for i in range(len(current_states)):

            # Если состояние последнее в эпизоде:
            if game_finished[i]:
                q_values[i, actions[i]] = rewards[i]
            # Если состояние не последнее в эпизоде:
            else:
                q_values[i, actions[i]] = rewards[i] + self.gamma * next_q_values[i, actions[i]]

        # Обучаем модель:
        self.main_model.fit(previous_states, q_values, batch_size=self.batch_size, verbose=0)

    @staticmethod
    def moving_average(data, width=10):
        """
        Функция для подсчета скользящего среднего всех значений
        Args:
          data — входной массив,
          width — длина, на которую считаем скользящее среднее
        Returns: результат свертки данных на фильтр из единиц — наше скользящее среднее
        """

        # Длина свертки:
        width = min(width, len(data))

        # Создадим паддинг для свертки:
        data = np.concatenate([np.repeat(data[0], width), data])

        # Возвращаем результат свертки:
        return (np.convolve(data, np.ones(width), 'valid') / width)[1:]

    def init_environment(self, load_pretrained=True):
        """
            load_pretrained: Решаем, если мы обучаем модель с нуля или продолжаем предыдущую сессию обучения
        """

        # Инициализируем среду:
        self.game = DoomGame()

        # Загружаем сценарий «Защищай центр»
        self.game.load_config(os.path.join(LIB_PATH, "scenarios", "defend_the_center.cfg"))

        # Устанавливаем размер кадров среды (будет 640×480):
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)

        # Нужно отключить окно, чтобы не было ошибки:
        self.game.set_window_visible(False)

        # Инициализируем игру:
        self.game.init()

        # Извлекаем количество возможных уникальных действий:
        self.action_size = self.game.get_available_buttons_size()

        # Создаем основную модель (будет обучаться):
        self.main_model = self.make_dqn()

        # Создаем целевую сеть (не будет обучаться, периодически будет обновляться под основную модель):
        self.target_model = self.make_dqn()

        # Устанавливаем параметры целевой модели (копируем в нее значения основной модели):
        self.update_target_model()

        # Если хотим продолжить текущее обучение, загружаем сохраненные веса для основной и целевой моделей:
        if load_pretrained:
            self.main_model.load_weights(MODEL_FILE)
            self.target_model.load_weights(MODEL_FILE)
            # Также загружаем ранее сохраненные статистики из pickle файла:
            with open(STATS_FILE, 'rb') as f:
                self.record_rewards, self.record_kills, self.record_ammos, self.episode_number, self.timestep, \
                self.epsilon = pickle.load(f)

        # Иначе мы просто инициализируем списки, в которых будет храниться статистика о работе агента:
        else:
            self.record_rewards = []  # Сюда будем записывать награды за 10 эпизодов (для анализа статистики)
            self.record_kills = []  # Сюда будем записывать количество убитых врагов (для анализа статистики)
            self.record_ammos = []  # Сюда будем записывать количество оставшихся патронов (для анализа статистики)
            self.episode_number = 1  # Инициализируем номер эпизода как 1
            self.timestep = 0  # Инициализируем номер шага как 0

    def game_loop(self):
        # Генерируем новый эпизод:
        self.game.new_episode()

        # Извлекаем первый кадр (это еще не полноценное состояние):
        game_data = self.game.get_state()

        # Извлекаем информацию об игре (количество убитых врагов, патроны, здоровье):
        current_info = game_data.game_variables

        # Записываем информацию о текущем моменте как 'предыдущий' момент (чтобы потом мы могли сравнить разницу):
        previous_info = current_info

        # Извлекаем кадр из игры (480×640):
        frame = game_data.screen_buffer
        # Предобрабатываем кадр в черно-белый (размер 64×64):
        processed_frame = self.preprocess_frame(frame)

        # В качестве первого состояния просто дублируем кадр 4 раза:
        current_state = np.stack([processed_frame.reshape(64, 64)] * self.num_frames, axis=2)
        # Инициализируем предыдущий шаг как текущий шаг:
        previous_state = current_state

        # Инициализируем награды:

        interval_reward = 0  # за интервал (10 эпизодов) как 0
        interval_kills = 0  # за количество убитых врагов (10 эпизодов) как 0
        interval_ammos = 0  # за количество оставшихся патронов (10 эпизодов) как 0

        # Обучение

        while self.episode_number < 1500:

            # Увеличиваем номер шага на 1:
            self.timestep += 1
            # Извлекаем индекс награды и новое значение эпсилон:
            action_index, self.epsilon = self.get_action(previous_state, self.epsilon)
            # Приводим награду в onehot массив:
            action_onehot = to_categorical(action_index)
            # Подаем действие в игровую среду в качестве списка:
            self.game.set_action(action_onehot.tolist())
            # Игра продвигается на 4 кадра (значение frames_per_action):
            self.game.advance_action(self.frames_per_action)

            # Предобрабатываем кадр в черно-белый (размер 64×64×1):
            game_data = self.game.get_state()

            # Проверяем, если эпизод закончился:
            episode_done = self.game.is_episode_finished()

            # Нам необходимо возобновить среду и записать нужные статистики когда заканчивается эпизод:
            if episode_done:
                print(
                    f"Закончился {self.episode_number}-й эпизод. Значение эпсилон: {round(self.epsilon, 2)}, Количество убитых врагов: {current_info[0]}, количество оставшихся патронов: {current_info[1]}")

                self.episode_number += 1  # Увеличиваем номер эпизода на 1:
                interval_kills += current_info[0]
                interval_ammos += current_info[1]

                # Чтобы не собирать слишком много данных и чтобы их было удобно отображать на графике

                # Записываем результат раз в 10 эпизодов:
                if self.episode_number % 10 == 0 and self.episode_number > 0:
                    # Добавляем награду в список всех наград:
                    self.record_rewards.append(interval_reward)
                    # Добавляем количество убитых врагов:
                    self.record_kills.append(interval_kills)
                    # Добавляем количество неиспользованных патронов:
                    self.record_ammos.append(interval_ammos)
                    # Записываем результаты в графики:
                    #show_scores(record_rewards, record_kills, record_ammos)

                    # Сохраняем веса модели:
                    self.main_model.save_weights(MODEL_FILE)

                    # Записываем статистику в файл через библиотеку pickle:
                    with open(STATS_FILE, 'wb') as f:
                        pickle.dump([self.record_rewards, self.record_kills, self.record_ammos, self.episode_number, self.timestep, self.epsilon], f)
                    print("Статистика успешно сохранена.")

                    # Заново инициализируем значения статистики для интервала в 10 эпизодов:
                    interval_reward, interval_kills, interval_ammos = 0, 0, 0

                    # Начинаем новый эпизод игры:
                self.game.new_episode()
                # Извлекаем новое состояние игры:
                game_data = self.game.get_state()

            # Извлекаем информацию об игровой среде (количество убитых врагов, неиспользованных патронов,
            # текущее здоровье):
            current_info = game_data.game_variables
            # Извлекаем новый кадр из игры:
            frame = game_data.screen_buffer
            # Предобрабатываем кадр (новая размерность будет 64×64×1):
            processed_frame = self.preprocess_frame(frame)
            # Обновляем состояние — удаляем последний кадр и добавляем новый:
            current_state = np.append(processed_frame, current_state[:, :, :self.num_frames - 1], axis=2)

            # Извлекаем награду за шаг из среды (логика, которую не можем менять):
            environment_reward = self.game.get_last_reward()
            # Извлекаем награду за шаг из самописной функции (самописная награда, значит, можем менять логику):
            custom_reward = self.get_reward(previous_info, current_info, episode_done)
            # Общая награда — это сумма награды из среды и самописной награды:
            reward = environment_reward + custom_reward

            # Добавляем награду в переменную для статистики:
            interval_reward += reward

            # Добавляем предыдущее состояние, действие, награду и текущее состояние в память:
            self.add_to_memory(previous_state, action_index, reward, current_state, episode_done)

            # Обучаем нашу модель раз в 100 шагов, но только если у нас достаточно данных в памяти:
            if self.timestep % self.timesteps_per_train == 0 and len(self.memory) > self.observation_steps:
                self.train_network()

            # Обновляем целевую модель весами основной модели раз в заданное количество (5 000) шагов:
            if self.timestep % self.target_model_update_frequency == 0:
                self.update_target_model()

            # Запоминаем предыдущую информацию:
            previous_info = current_info
            # Запоминаем предыдущее состояние:
            previous_state = current_state


if __name__ == "__main__":
    vizdoom_model = ViZDoomModel()

    # Создание игровой среды, инициализация нейронной сети
    vizdoom_model.init_environment(load_pretrained=True)

    # Создание игрового цикла и обучение
    vizdoom_model.game_loop()
