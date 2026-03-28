import numpy as np
import stable_baselines3 as sb
from stable_baselines3.common.env_util import make_vec_env

from data_loaders import SyntheticData
from rs_algorithm.models.user_models.user_01_dpr import DPRUserModel
from rs_algorithm.system2.environments.env_03_simple import SimpleEnvironment


def test_dpr_user_model():
    data = SyntheticData(d_embeds=1, n_clusters=2)

    item_0 = np.array([[+1]])
    item_1 = np.array([[-1]])

    user = DPRUserModel(data, dissimilarity_reward=2)
    user.preference = item_0

    # Giving user similar items
    for _ in range(10):
        user.react(None, None, item_0)
        print(user.fatigue)
    assert user.fatigue_buildup == 10.0

    # Giving user dissimilar items
    for _ in range(5):
        user.react(None, None, item_1)
        print(user.fatigue)
    assert user.fatigue_buildup == 0.0


def test_simple():
    data = SyntheticData()
    user = DPRUserModel(data)
    env = SimpleEnvironment(data, user)

    obs, _ = env.reset()

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(reward, info)
        if done or truncated:
            break


def test_with_stable_baselines3():
    data = SyntheticData()

    def create_user():
        user = DPRUserModel(data)
        user.preference = data.get_items_embeds(0)[None, :]
        return user

    env = make_vec_env(lambda: SimpleEnvironment(data, create_user()), n_envs=1)

    # model = sb.DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=10000, learning_starts=100, batch_size=32, )
    model = sb.PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=1024, batch_size=64, gamma=0.99, ent_coef=0.01)
    model.learn(total_timesteps=30000)

    env = SimpleEnvironment(data, create_user())

    obs, _ = env.reset()

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(reward, info)
        if done or truncated:
            break
