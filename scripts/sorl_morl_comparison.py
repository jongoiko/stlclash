from typing import Any

import morl_baselines
import numpy as np
import yaml
from rlrom.testers import RLTester
from rlrom.trainers import RLTrainer

import stlclash.tasks


def get_config(
    task: stlclash.tasks.Task,
    algo_config: dict,
    train_config: dict,
    training_timesteps: int,
    eval_episodes: int,
    num_weight_trials: int,
    seed: int,
) -> dict:
    with open(task.specs_yaml_path, "r") as f:
        task_name = task.id.split("/")[1]
        specs_config = yaml.safe_load(f)
    config = {
        "env_name": task.env_name,
        "import_module": "stlclash",
        "model_name": task.id,
        "model_use_specs": True,
        "cfg_specs": specs_config,
        "this_cfg_pathdir": str(task.specs_yaml_path.parent),
        "cfg_train": {
            "total_timesteps": training_timesteps,
            "algo": algo_config,
            "train_kwargs": train_config,
        },
        "cfg_test": {"init_seeds": seed, "num_ep": eval_episodes // num_weight_trials},
    }
    # Monitor the same formulas that we use for training.
    config["cfg_specs"]["eval_formulas"] = {
        formula_name: {"eval_all_steps": False}
        for formula_name in list(config["cfg_specs"]["reward_formulas"].keys())
        + [f"task_{task_name}"]
    }
    return config


def train_conjunction_policy(
    task: stlclash.tasks.Task,
    algo_config: dict,
    train_config: dict,
    training_timesteps: int,
    eval_episodes: int,
    seed: int = 42,
) -> tuple[dict, Any]:
    config = get_config(
        task, algo_config, train_config, training_timesteps, eval_episodes, 1, seed
    )
    task_name = task.id.split("/")[1]
    config["cfg_specs"]["reward_formulas"] = {f"task_{task_name}": {"past_horizon": 0}}
    config["cfg_specs"]["multi_objective"] = False
    config["cfg_test"]["num_ep"] = eval_episodes
    trainer = RLTrainer(config)
    model = trainer.train()
    tester = RLTester(config)
    tester.model = model
    test_result = tester.run_cfg_test(reload_model=False)
    res_all_ep = test_result["res_all_ep"]
    eval_formulas = tester.env.get_wrapper_attr("eval_formulas")
    sats = {}
    for f_name in eval_formulas.keys():
        ratio_sat = res_all_ep["eval_formulas"][f_name]["ratio_init_sat"]
        sats[f_name] = ratio_sat
    return sats, model


def mo_policy_search(
    task: stlclash.tasks.Task,
    algo_config: dict,
    train_config: dict,
    training_timesteps: int,
    eval_episodes: int,
    num_weight_trials: int,
    seed: int = 42,
) -> tuple[dict, Any]:
    assert num_weight_trials <= eval_episodes
    assert eval_episodes % num_weight_trials == 0
    assert training_timesteps % num_weight_trials == 0
    config = get_config(
        task,
        algo_config,
        train_config,
        training_timesteps,
        eval_episodes,
        num_weight_trials,
        seed,
    )
    # The multi-objective policy is only trained once
    trainer = RLTrainer(config)
    model = trainer.train()
    tester = RLTester(config)
    weights = np.atleast_2d(
        morl_baselines.common.weights.random_weights(
            task.get_num_objectives(), num_weight_trials, seed=seed
        )
    )
    satisfaction_rates: dict = {}
    for weights_trial in weights:
        # model.weights = weights_trial
        tester.model = model
        tester.model_action_function = lambda model, obs: model.eval(obs, weights_trial)
        test_result = tester.run_cfg_test(reload_model=False)
        res_all_ep = test_result["res_all_ep"]
        eval_formulas = tester.env.get_wrapper_attr("eval_formulas")
        for f_name in eval_formulas.keys():
            ratio_sat = res_all_ep["eval_formulas"][f_name]["ratio_init_sat"]
            prev_sat = satisfaction_rates.get(f_name, (0, None))[0]
            if ratio_sat > prev_sat:
                satisfaction_rates[f_name] = ratio_sat, weights_trial
    return satisfaction_rates, model


def so_policy_search(
    task: stlclash.tasks.Task,
    algo_config: dict,
    train_config: dict,
    training_timesteps: int,
    eval_episodes: int,
    num_weight_trials: int,
    seed: int = 42,
) -> dict:
    assert num_weight_trials <= eval_episodes
    assert eval_episodes % num_weight_trials == 0
    assert training_timesteps % num_weight_trials == 0
    config = get_config(
        task,
        algo_config,
        train_config,
        training_timesteps,
        eval_episodes,
        num_weight_trials,
        seed,
    )
    config["cfg_specs"]["multi_objective"] = False
    config["cfg_train"]["total_timesteps"] = training_timesteps // num_weight_trials
    weights = np.atleast_2d(
        morl_baselines.common.weights.random_weights(
            task.get_num_objectives(), num_weight_trials, seed=seed
        )
    )
    satisfaction_rates: dict = {}
    for weights_trial in weights:
        for weight, formula_name in zip(
            weights_trial, config["cfg_specs"]["reward_formulas"]
        ):
            config["cfg_specs"]["reward_formulas"][formula_name]["weight"] = float(
                weight
            )
        trainer = RLTrainer(config)
        model = trainer.train()
        tester = RLTester(config)
        tester.model = model
        test_result = tester.run_cfg_test(reload_model=False)
        res_all_ep = test_result["res_all_ep"]
        eval_formulas = tester.env.get_wrapper_attr("eval_formulas")
        for f_name in eval_formulas.keys():
            ratio_sat = res_all_ep["eval_formulas"][f_name]["ratio_init_sat"]
            prev_sat = satisfaction_rates.get(f_name, (0, None))[0]
            if ratio_sat > prev_sat:
                satisfaction_rates[f_name] = ratio_sat, weights_trial
        del trainer, tester
    return satisfaction_rates
