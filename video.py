import os.path

from main import *

VIDEO_FILE = os.path.join(CONTENT_PATH, 'episode.avi')


class VideoDoom(ViZDoomModel):

    def __init__(self):
        super().__init__()

    def make_video(self):
        # Кадры из игр будут записываться в этот массив:
        video_frames = []
        # Устанавливаем эпсилон как 0 (после обучения):
        epsilon = 0

        # Снова инициализируем среду:
        game = DoomGame()
        # Загружаем сценарий «Защищай центр»:
        game.load_config(os.path.join(LIB_PATH, "scenarios", "defend_the_center.cfg"))
        # Устанавливаем размер кадров среды (будет 640×480):
        game.set_screen_resolution(ScreenResolution.RES_640X480)
        # Нужно отключить окно, чтобы не было ошибки:
        game.set_window_visible(False)
        # Снова инициализируем игру:
        game.init()

        # Извлекаем количество возможных уникальных действий:
        self.action_size = game.get_available_buttons_size()

        # Создаем основную модель (будет управлять агентом):
        self.main_model = self.make_dqn()
        self.main_model.load_weights(MODEL_FILE)

        # Генерируем новый эпизод:
        game.new_episode()
        # Извлекаем первый кадр (это еще не полноценное состояние):
        game_data = game.get_state()

        # Извлекаем кадр из игры (480×640):
        frame = game_data.screen_buffer
        # Предобрабатываем кадр в черно-белый (размер 64×64):
        processed_frame = self.preprocess_frame(frame)
        # В качестве первого состояния просто дублируем кадр 4 раза:
        current_state = np.stack([processed_frame.reshape(self.image_width, self.image_height)] * self.num_frames,
                                 axis=2)
        # Записываем текущее состояние в предыдущее состояние:
        previous_state = current_state

        while True:
            # Извлекаем индекс награды и новое значение эпсилон:
            action_index, epsilon = self.get_action(current_state, epsilon)
            # Приводим награду в onehot-массив:
            action_onehot = to_categorical(action_index)
            # Подаем действие в игровую среду в качестве списка:
            game.set_action(action_onehot.tolist())
            # Игра продвигается на 4 кадра (значение frames_per_action):
            game.advance_action(self.frames_per_action)

            # Предобрабатываем кадр в черно-белый (размер 64×64×1):
            game_data = game.get_state()

            # Проверяем, если эпизод закончился:
            episode_done = game.is_episode_finished()

            # Нам необходимо возобновить среду и записать нужные статистики, когда заканчивается эпизод:
            if episode_done:
                # Затем необходимо начать новый эпизод игры:
                game.new_episode()
                # Извлекаем новое состояние игры:
                game_data = game.get_state()
                # Выходим из игрового цикла:
                break

            # Извлекаем новый кадр из игры:
            frame = game_data.screen_buffer
            # Добавляем кадр в массив, меняем формат размерности (3, width, height) -> (width, height, 3):
            video_frames.append(np.rollaxis(frame, 0, 3))

            # Предобрабатываем кадр (новая размерность будет 64×64×1):
            processed_frame = self.preprocess_frame(frame)
            # Обновляем состояние — удаляем последний кадр и добавляем новый:
            current_state = np.append(processed_frame, current_state[:, :, :self.num_frames - 1], axis=2)

            # Запоминаем предыдущее состояние:
            previous_state = current_state

        # Чем больше кадров в секунду, тем быстрее будет проигрываться видео
        out = cv2.VideoWriter(VIDEO_FILE, cv2.VideoWriter_fourcc(*'DIVX'), 20, (640, 480))

        # В цикле добавляем каждый кадр в видео (делаем предобработку кадра — меняем каналы с RGB в BGR,
        # поскольку CV2 воспринимает каналы как BGR):
        for i in range(len(video_frames)):
            out.write(cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR))

        # Закрываем объект для создания видео
        out.release()

        print(f'Видео находится в файле: {VIDEO_FILE}')


if __name__ == "__main__":
    videodoom = VideoDoom()
    videodoom.make_video()
