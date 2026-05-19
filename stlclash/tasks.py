from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Task:
    id: str
    env_name: str
    specs_yaml_path: Path

    def get_num_objectives(self) -> int:
        with open(self.specs_yaml_path, "r") as f:
            specs = yaml.load(f, Loader=yaml.SafeLoader)
        return len(specs["reward_formulas"])


TASKS: list[Task] = []


def register_tasks(env_name: str, prefix: str, specs_path: Path) -> None:
    for spec_path in specs_path.rglob(prefix + "_*.yml"):
        spec_name = spec_path.stem.replace(prefix + "_", "")
        TASKS.append(Task(f"{prefix}/{spec_name}", env_name, spec_path.resolve()))
