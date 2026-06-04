import pickle

import pandas as pd

import stlclash
from scripts.sorl_morl_comparison import mo_policy_search
from scripts.sorl_morl_comparison import so_policy_search
from scripts.sorl_morl_comparison import train_conjunction_policy


DEFAULT_TRAINING_STEPS = 100_000
DEFAULT_EVAL_EPISODES = 1000
DEFAULT_WEIGHT_TRIALS = 20
DEFAULT_SEED = 42


def get_task_settings() -> pd.DataFrame:
    task_settings = {}
    for task in stlclash.TASKS:
        task_settings[task.id] = {
            "training_steps": DEFAULT_TRAINING_STEPS,
            "eval_episodes": DEFAULT_EVAL_EPISODES,
            "weight_trials": DEFAULT_WEIGHT_TRIALS,
        }
        if any(
            name in task.id
            for name in [
                "three_states/",
                "four_states_circular/",
                "long_line/",
                "loop_long_line/",
            ]
        ):
            task_settings[task.id]["eval_episodes"] = task_settings[task.id][
                "weight_trials"
            ]
    algo_registry = {
        "sbx_crossq": {
            "type": "single",
            "conf": {"seed": DEFAULT_SEED},
            "train_kwargs": {},
        },
        "sbx_sac": {
            "type": "single",
            "conf": {"policy": "SimbaPolicy", "seed": DEFAULT_SEED},
            "train_kwargs": {},
        },
        "qlearning": {
            "type": "single",
            "conf": {},
            "train_kwargs": {},
        },
        "capql": {
            "type": "multi",
            "conf": {"seed": DEFAULT_SEED},
            "train_kwargs": {},
        },
        "mpmoql": {
            "type": "multi",
            "conf": {"seed": DEFAULT_SEED},
            "train_kwargs": {
                "timesteps_per_iteration": 2_000,
            },
        },
        "gpils_continuous_action_jax": {
            "type": "multi",
            "conf": {"seed": DEFAULT_SEED},
            "train_kwargs": {},
        },
    }
    rows = []
    for task in stlclash.TASKS:
        task_id = task.id
        tcfg = task_settings.get(task_id, {})
        training_steps = tcfg.get(
            "training_steps",
            DEFAULT_TRAINING_STEPS,
        )
        eval_episodes = tcfg.get(
            "eval_episodes",
            DEFAULT_EVAL_EPISODES,
        )
        weight_trials = tcfg.get(
            "weight_trials",
            DEFAULT_WEIGHT_TRIALS,
        )
        for algo, cfg in algo_registry.items():
            assert isinstance(cfg["conf"], dict)
            assert isinstance(cfg["train_kwargs"], dict)
            conf = cfg["conf"].copy()
            train = cfg["train_kwargs"].copy()
            append = True
            if "tiny" in str(task.specs_yaml_path):
                if cfg["type"] == "single" and algo != "qlearning":
                    append = False
                if cfg["type"] == "multi" and (algo != "mpmoql" and algo != "pql"):
                    append = False
            else:
                if cfg["type"] == "single" and algo == "qlearning":
                    append = False
                if cfg["type"] == "multi" and (algo == "mpmoql" or algo == "pql"):
                    append = False
            if cfg["type"] == "multi" and train is not None:
                train["ref_point"] = [-100] * task.get_num_objectives()
            if cfg["type"] == "multi" and algo == "mpmoql":
                train["timesteps_per_iteration"] = training_steps // weight_trials
            if append:
                rows.append(
                    {
                        "task": task_id,
                        "algorithm": algo,
                        "algo_type": cfg["type"],
                        "training_steps": training_steps,
                        "eval_episodes": eval_episodes,
                        "weight_trials": weight_trials,
                        "seed": DEFAULT_SEED,
                        "conf": conf,
                        "train_kwargs": train,
                    }
                )
    return pd.DataFrame(rows).set_index(["task", "algorithm"])


def main() -> None:
    df = get_task_settings()
    task_map = {task.id: task for task in stlclash.TASKS}
    results = []
    for (task_id, algo), row in df.iterrows():  # type: ignore
        task = task_map[task_id]
        print(f"\n{'=' * 80}\n{algo} | {task_id}\n{'=' * 80}")
        try:
            conj_sats, _ = None, None
            if row["algo_type"] == "single":
                sats = so_policy_search(
                    task=task,
                    algo_config={algo: row["conf"]},
                    train_config=row["train_kwargs"],
                    training_timesteps=row["training_steps"],
                    eval_episodes=row["eval_episodes"],
                    num_weight_trials=row["weight_trials"],
                    seed=row["seed"],
                )

                conj_sats, _ = train_conjunction_policy(
                    task=task,
                    algo_config={algo: row["conf"]},
                    train_config=row["train_kwargs"],
                    training_timesteps=row["training_steps"],
                    eval_episodes=row["eval_episodes"],
                    seed=row["seed"],
                )
            else:
                sats, _ = mo_policy_search(
                    task=task,
                    algo_config={algo: row["conf"]},
                    train_config=row["train_kwargs"],
                    training_timesteps=row["training_steps"],
                    eval_episodes=row["eval_episodes"],
                    num_weight_trials=row["weight_trials"],
                    seed=row["seed"],
                )
            result_row = {
                "task": task_id,
                "algorithm": algo,
                "algo_type": row["algo_type"],
                "training_steps": row["training_steps"],
                "eval_episodes": row["eval_episodes"],
                "weight_trials": row["weight_trials"],
                "seed": row["seed"],
                "mo_sats": sats,
                "conj_sats": conj_sats,
            }
            results.append(result_row)
        except Exception as e:
            print(f"FAILED: {algo} | {task_id}")
            print(e)
            results.append(
                {
                    "task": task_id,
                    "algorithm": algo,
                    "error": str(e),
                }
            )
            break
    with open("results.pickle", "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
