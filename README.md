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
   - Apply actions - `_pre_physics_step()` applies actions once per RL step prior to physics steps, `_apply_action()` called decimation no. of times per RL step prior to physics step. Multiple physics steps can occur per action taken by a policy. **_In simple words: `pre_physics_step()` is called every sim step; `_apply_action()` is called only after every RL step when policy outputs an action._**

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

## Adding your Robots

### Create robots

New robots can be added to the project in separate files.

1. Create a directory called  `robots` in `isaac_lab_tutorial/source/isaac_lab_tutorial/isaac_lab_tutorial`.
2. Inside the `robots` folder, create the `__init__.py` file first.
3. Then create new files for separate robots that need to be added in the project. Eg: `jetbot.py`.

A minimal way to create a robot file is as follows:

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)
```

### Modify Env Cfg File

Import the robot inside the **Env Cfg** file as follows:

```python
from isaac_lab_tutorial.robots.jetbot import JETBOT_CONFIG
```

Then replace the `robot_cfg` variable in the **Env Cfg** class such as the following example. Also modify the `action_space`, `observation_space`, `state_space` and `dof_names` variables.

```python
@configclass
class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 2
    observation_space = 3
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=4.0, replicate_physics=True)
    dof_names = ["left_wheel_joint", "right_wheel_joint"]
```

Also modify the `scene` parameters such as `num_envs` and `env_spacing`.

### Modify Env file

1. Modify the `__init__()` function by specifying the correct joints to find from the robot.
2. Modify `_setup_scene()` if required.
3. Modify the `_pre_physics_step()` and `_apply_action()` functions. Any action that needs to be applied at every sim step is added in `_pre_physics_step()`. The output output by the decision-making RL policy is added in `_apply_action()` function.
4. Modify `_get_observations()` and `_get_rewards()`. We can get observations through ```self.robot.data```.
5. Add termination and reset conditions in `_get_dones()` and `_reset_idx()` functions.

More information about adding different robot assets to the project can be found [in this tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/add_new_robot.html#tutorial-add-new-robot).

## Adding Visualization Markers

1. Define visualization markers by adding the following code at the top of the **Env*** file (in global scope):

```python
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)
```

2. Then modify the `_setup_scene()` function to include the markers. Eg: add the following code to add markers for both commands and robot's forward vel:

```python
def _setup_scene(self):

    # ... existing code ...

    self.visualization_markers = define_markers()

    # setting aside useful variables for later
    self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
    self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
    self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
    self.commands[:,-1] = 0.0
    self.commands = self.commands/torch.linalg.norm(self.commands, dim=1, keepdim=True)

    # offsets to account for atan range and keep things on [-pi, pi]
    ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
    gzero = torch.where(self.commands > 0, True, False)
    lzero = torch.where(self.commands < 0, True, False)
    plus = lzero[:,0]*gzero[:,1]
    minus = lzero[:,0]*lzero[:,1]
    offsets = torch.pi*plus - torch.pi*minus
    self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

    self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
    self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
    self.marker_offset[:,-1] = 0.5
    self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
    self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()

```
Read [this tutorial](https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/training_jetbot_gt.html) to understand how to calculate and create marker directions.

3. Add function to visualize the markers and call the function in the `_pre_physics_step()` function

```python
def _visualize_markers(self):
    # get marker locations and orientations
    self.marker_locations = self.robot.data.root_pos_w
    self.forward_marker_orientations = self.robot.data.root_quat_w
    self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

    # offset markers so they are above the jetbot
    loc = self.marker_locations + self.marker_offset
    loc = torch.vstack((loc, loc))
    rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

    # render the markers
    all_envs = torch.arange(self.cfg.scene.num_envs)
    indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
    self.visualization_markers.visualize(loc, rots, marker_indices=indices)
```

4. Add code to reset markers inside `_reset_idx()`.

```python
def _reset_idx(self, env_ids: Sequence[int] | None):
    if env_ids is None:
        env_ids = self.robot._ALL_INDICES
    super()._reset_idx(env_ids)

    # pick new commands for reset envs
    self.commands[env_ids] = torch.randn((len(env_ids), 3)).cuda()
    self.commands[env_ids,-1] = 0.0
    self.commands[env_ids] = self.commands[env_ids]/torch.linalg.norm(self.commands[env_ids], dim=1, keepdim=True)

    # recalculate the orientations for the command markers with the new commands
    ratio = self.commands[env_ids][:,1]/(self.commands[env_ids][:,0]+1E-8)
    gzero = torch.where(self.commands[env_ids] > 0, True, False)
    lzero = torch.where(self.commands[env_ids]< 0, True, False)
    plus = lzero[:,0]*gzero[:,1]
    minus = lzero[:,0]*lzero[:,1]
    offsets = torch.pi*plus - torch.pi*minus
    self.yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

    # set the root state for the reset envs
    default_root_state = self.robot.data.default_root_state[env_ids]
    default_root_state[:, :3] += self.scene.env_origins[env_ids]

    self.robot.write_root_state_to_sim(default_root_state, env_ids)
    self._visualize_markers()
```

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