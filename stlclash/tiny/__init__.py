from pathlib import Path

from gymnasium.envs.registration import register

from stlclash.tasks import register_tasks

register(
    "three-states-v0",
    entry_point="stlclash.tiny.straight_line:make_straight_line_env",
    max_episode_steps=10,
    kwargs=dict(
        num_states=3,
        circular=False,
        start_location=1,
    ),
)
register_tasks("three-states-v0", "three_states", Path(__file__).parent.resolve())

register(
    "four-states-circular-v0",
    entry_point="stlclash.tiny.straight_line:make_straight_line_env",
    max_episode_steps=24,
    kwargs=dict(
        num_states=4,
        circular=True,
        start_location=1,
    ),
)
register_tasks(
    "four-states-circular-v0", "four_states_circular", Path(__file__).parent.resolve()
)

register(
    "long-line-v0",
    entry_point="stlclash.tiny.straight_line:make_straight_line_env",
    max_episode_steps=24,
    kwargs=dict(
        num_states=20,
        circular=False,
        start_location=2,
    ),
)
register_tasks("long-line-v0", "long_line", Path(__file__).parent.resolve())

register(
    "long-line-loop-v0",
    entry_point="stlclash.tiny.straight_line:make_straight_line_env",
    max_episode_steps=24,
    kwargs=dict(
        num_states=20,
        circular=True,
        start_location=2,
    ),
)
register_tasks("long-line-loop-v0", "loop_long_line", Path(__file__).parent.resolve())
