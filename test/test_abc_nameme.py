from rs_algorithm.system2.models import SEWADQNModelFactory
from rs_algorithm.system2.environments import SEWADQNEnvironment


def test_huh():
    print("Testing HUH")

    env = SEWADQNEnvironment()
    factory = SEWADQNModelFactory()
    model = factory.create_model(env)
    model.learn(total_timesteps=100)

    obs, _ = env.reset()

    for _ in range(10):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

        print(f"Recommended item: {action}, Reward: {reward:.3f}")

        if done:
            break
