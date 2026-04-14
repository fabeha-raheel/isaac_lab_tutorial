# Isaac_lab_Tutorial

This project has been created by following the walkthrough examples from Isaaclab documentation found at: [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html)

## Create a Project

Install IsaacLab using the installation instructions.

Activate conda environment:

```bash
conda activate env_isaaclab
```

Create an External project using template generator:

```bash
cd ~/IsaacLab

./isaaclab.sh --new  # or "./isaaclab.sh -n"
```

Set the following:

- Task type: External
- Project path: <Just press enter - leave home directory for now>
- Project name: isaac_lab_tutorial
- Isaac Lab workflow: Direct | single-agent
- RL library: rsl_rl

The project will be created.

The template creates the project as a python package which needs to be installed so that it is registered with Isaac Lab.

Install the package using the following commands:

```bash
cd ~/isaac_lab_tutorial
python -m pip install -e source/isaac_lab_tutorial/
```

Check if installation is successful by listing the envs:

```bash
python -m pip install -e source/isaac_lab_tutorial/
```

To understand project structure, see this link: [Project Structure](https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/project_structure.html)

## Creating RL Tasks

The two most important files for defining the RL learning tasks are:

1. The **Env Cfg** file (inherits from DirectRLEnvCfg) - can be found at `~/isaac_lab_tutorial/source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/direct/isaac_lab_tutorial/isaac_lab_tutorial_env_cfg.py`
   - 3 class instances defined here: **sim** (SimulationCfg), **scene** (InteractiveSceneCfg) and **robot** (ArticulationCfg).
   - add envconfiguration parameters - can be used from existing templates
   - add sizes of `action_space`, `observation_space` and `state_space`
   - also add task-specific attributes such as scaling for reward terms, thresholds for reset conditions
2. The **Env** file (inherits from DirectRLEnv) - can be found at `~/isaac_lab_tutorial/source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/direct/isaac_lab_tutorial/isaac_lab_tutorial_env.py`
   - include Env Cfg (which was defined above)
   - define functions required for task such as applying actions, computing resets, rewards or observations
   - Create scene using `_setup_scene(self)` method
   - Define rewards in `_get_rewards(self)` and `computer_rewards()` - get_rewards calls compute_rewards(), compute_rewards() is where all the reward functions are implemented
   - compute observation buffer in the `_get_observations(self)` method - output should return a dict with "policy" as key and full obs buffer as value. Eg. ```observations = {"policy": obs}```
   - Reset envs using `_get_dones(self)` and `_reset_idx()` methods - get reset env_ids (time_out and out_of_bounds) in _get_dones(), implement reset logic in _reset_idx()
   - Apply actions - `_pre_physics_step()` applies actions once per RL step prior to physics steps, `_apply_action()` called decimation no. of times per RL step prior to physics step

## Add Domain Randomization (DR)

DR is specified using by creating an `EventCfg` class.

Each randomization is configured as an `EventTerm` object that includes:

- `func` param to specify function to call during randomization; they are found in the `events` module
- `mode` param which can be `"startup"`, `"reset"` or `"interval"`
- `params` dictionary to provide necessary args to `func`

After setting up `EventCfg`, it needs to be added in the **Env Cfg** file (eg: `isaac_lab_tutorial_env_cfg.py`) and assigned to the variable `events` as:

```python
class Env_Cfg:
    events: EventCfg = EventCfg()
```

See [example](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html#tutorial-create-direct-rl-env)

## Add Action Noise and Observation Noise

These are added in main task config **Env Cfg** using the `action_noise_model` and `observation_noise_model` variables as follows:

```python

@configclass
class MyTaskConfig:

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
      noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
      bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
      noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
      bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )
```

See [more details](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html#tutorial-create-direct-rl-env)

## Additional Points

Add the following in .gitignore to prevent .vscode folder from being pushed into GitHub:

```bash
# vscode
.vscode/
```

---

<!-- # Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/isaac_lab_tutorial

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/isaac_lab_tutorial/isaac_lab_tutorial/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/isaac_lab_tutorial"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
... 
```-->