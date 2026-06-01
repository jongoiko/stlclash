from pathlib import Path

from gymnasium.envs.registration import register

from stlclash.tasks import register_tasks

register_tasks("Pendulum-v1", "pendulum", Path(__file__).parent.resolve())

register(
    "ContinuousMountainCar-v0",
    entry_point="stlclash.classic_control.mountain_car:make_mountain_car_env",
    max_episode_steps=500,
)
register_tasks(
    "ContinuousMountainCar-v0", "mountain_car", Path(__file__).parent.resolve()
)
