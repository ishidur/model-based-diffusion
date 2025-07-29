import os
from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, html
from etils import epath
import jax
from jax import numpy as jnp
import mujoco

import mbd


class ExcavatorPose(PipelineEnv):
    def __init__(self):
        mj = mujoco.MjModel.from_xml_path(f"{mbd.__path__[0]}/assets/zx120/zx120.xml")
        sys = mjcf.load_model(mj)
        super().__init__(sys=sys, backend="mjx")

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -0.01, 0.01
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=-0.01, maxval=0.01
        )
        qvel = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=low, maxval=hi)
        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jnp.zeros(self.sys.act_size()))
        reward, done = jnp.zeros(2)
        return State(pipeline_state, obs, reward, done, {})

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state, action)
        reward = self._get_reward(pipeline_state)

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd], axis=-1)

    def _get_reward(self, pipeline_state: base.State) -> jax.Array:
        bucket_end = pipeline_state.sensordata[:7]
        target = pipeline_state.sensordata[7:]
        return -jnp.sum(jnp.abs(bucket_end[:3] - target[:3]))


def main():
    env = ExcavatorPose()
    rng = jax.random.PRNGKey(1)
    env_step = jax.jit(env.step)
    env_reset = jax.jit(env.reset)
    state = env_reset(rng)
    rollout = [state.pipeline_state]
    for _ in range(200):
        rng, rng_act = jax.random.split(rng)
        act = jax.random.uniform(rng_act, (env.action_size,), minval=-1.0, maxval=1.0)
        state = env_step(state, act)
        rollout.append(state.pipeline_state)
    webpage = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout)
    path = f"{mbd.__path__[0]}/../results/excavatorpose"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"{path}/vis.html", "w") as f:
        f.write(webpage)


if __name__ == "__main__":
    main()
