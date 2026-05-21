from pathlib import Path

from stlclash.tasks import register_tasks

register_tasks("Pendulum-v1", "pendulum", Path(__file__).parent.resolve())
