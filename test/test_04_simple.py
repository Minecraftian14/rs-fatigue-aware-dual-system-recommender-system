from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from pytest import raises, approx, mark

import stable_baselines3 as sb
from sklearn.metrics.pairwise import cosine_similarity
from stable_baselines3.common.env_util import make_vec_env

from data_loaders import SyntheticData
from rs_algorithm.models.user_models.blueprint_user_model import UserModel
from rs_algorithm.models.user_models.user_02_sim import SimUserModel
from rs_algorithm.system2.environments.env_04_simple import SimpleEnvironment

from miscellaneous import plot_data, angle, random_axis


def test_sim_user_model():
    item_0 = np.array([[+1]])
    item_1 = np.array([[-1]])

    user: UserModel = SimUserModel(preference=item_0,
                                   good_item_fatigue_weight=0.2,
                                   bad_item_fatigue_weight=0.1,
                                   good_item_discontent_weight=0.05,
                                   bad_item_discontent_weight=0.025)

    today = datetime.now()
    tomorrow = today + timedelta(days=1)

    # Giving user similar items
    for _ in range(5):
        user.react(today, None, item_0)
    assert user.fatigue == approx(1.0)
    assert user.is_fatigued == True

    with raises(Exception):
        user.react(today, None, item_0)

    # Giving user dissimilar items
    for _ in range(7):
        user.react(tomorrow, None, item_1)
    assert user.fatigue == approx(0.0)
    for _ in range(33):
        user.react(tomorrow, None, item_1)
    assert user.discontent == approx(1.0)

    with raises(Exception):
        user.react(today, None, item_1)


@mark.skip(reason="Not a test")
def test_orthogonality():
    data = SyntheticData(100000)
    pref = data.get_items_embeds()[None, np.random.randint(0, data.get_n_items())]
    print(cosine_similarity(pref, data.get_items_embeds()).mean())


def test_simple():
    data = SyntheticData(1000, n_clusters=100)
    while True:
        pref = data.get_items_embeds(np.random.randint(data.get_n_items()))[None, :]
        cd = cosine_similarity(pref, data.get_items_embeds()).mean()
        print('cd', cd)
        if cd < 0: break

    user = SimUserModel(pref, 0.3)
    env = SimpleEnvironment(data, user)

    axes = random_axis(data.d_item_embeds, 3)

    obs, _ = env.reset()

    rewards, fatigue, discontent = [], [], []
    d_axes = [[] for _ in axes]

    for _ in range(15):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        fatigue.append(info["fatigue"])
        discontent.append(info["discontent"])
        preference = info["preference"]
        for axis, d_axis in zip(axes, d_axes):
            d_axis.append(angle(axis, preference))
        if done or truncated: break

    axs = plt.subplots(2, 2)[1].ravel()
    plot_data(rewards, "Rewards", ax=axs[0])
    plot_data(fatigue, "Fatigue", ylim=(0, 1), ax=axs[1])
    plot_data(discontent, "Discontent", ylim=(0, 1), ax=axs[2])
    for d_axis in d_axes: plot_data(d_axis, ax=axs[3])
    plot_data(title="Preference", ylim=(0, np.pi), ax=axs[3])
    plt.show()


def test_with_stable_baselines3():
    data = SyntheticData()
    while True:
        pref = data.get_items_embeds(np.random.randint(data.get_n_items()))[None, :]
        cd = cosine_similarity(pref, data.get_items_embeds()).mean()
        print('cd', cd)
        if cd < 0: break

    def create_user():
        return SimUserModel(pref)

    env = make_vec_env(lambda: SimpleEnvironment(data, create_user()), n_envs=14)

    # model = sb.DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=10000, learning_starts=100, batch_size=32, )
    model = sb.PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=1024, batch_size=64, gamma=0.99, ent_coef=0.01)
    model.learn(total_timesteps=300000)
    model.save("temp/model_04_simple_sb3")

    env = SimpleEnvironment(data, create_user(), max_steps=1000)

    axes = random_axis(data.d_item_embeds, 3)

    obs, _ = env.reset()

    rewards, fatigue, discontent = [], [], []
    d_axes = [[] for _ in axes]

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        fatigue.append(info["fatigue"])
        discontent.append(info["discontent"])
        preference = info["preference"]
        for axis, d_axis in zip(axes, d_axes):
            d_axis.append(angle(axis, preference))
        if done or truncated: break

    axs = plt.subplots(2, 2)[1].ravel()
    plot_data(rewards, "Rewards", ax=axs[0])
    plot_data(fatigue, "Fatigue", ylim=(0, 1), ax=axs[1])
    plot_data(discontent, "Discontent", ylim=(0, 1), ax=axs[2])
    for d_axis in d_axes: plot_data(d_axis, ax=axs[3])
    plot_data(title="Preference", ylim=(0, np.pi), ax=axs[3])
    plt.show()
