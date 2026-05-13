from pathlib import Path

from gymnasium.envs.registration import register

from stlclash.tasks import register_tasks

register(
    "three-states-v0",
    entry_point="stlclash.tiny.three_states:make_three_states_env",
    max_episode_steps=10,
)
register_tasks("three-states-v0", "three_states", Path("./"))
