from rs_algorithm.system2.models import ABCNameMeModel
from rs_algorithm.system2.environments import ABCNameMeEnv


def test_huh():
    print("Testing HUH")

    env = ABCNameMeEnv()
    factory = ABCNameMeModel()
    model = factory.create_model(env)
    model.learn(total_timesteps=100)

    obs, _ = env.reset()

    for _ in range(10):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

        print(f"Recommended item: {action}, Reward: {reward:.3f}")

        if done:
            break
