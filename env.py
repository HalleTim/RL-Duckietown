from gym_duckietown.simulator import Simulator
from gymnasium import wrappers

def launchEnv():
    env = Simulator(
        seed=123, # random seed
        map_name="loop_empty",
        max_steps= 500001, # we don't want the gym to reset itself
        domain_rand=0,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=4, # start close to straight
        full_transparency=True,
        distortion=True,
    )
    env=integrateWrappers(env)

    return env

def integrateWrappers(env):
    env=wrappers.ResizeObservation(env,(60,80))
    env=wrappers.GrayscaleObservation(env, True)
    #env=wrappers.RescaleObservation(env, 0,1)
    env=wrappers.NormalizeObservation(env)
    env=wrappers.TransformObservation(env, lambda obs: obs.transpose(2,0,1) , env.observation_space)
    env=wrappers.TransformReward(env, lambda r: -10 if r==-1000 else r+10 if r>0 else r+4)
    env=wrappers.TransformAction(env,lambda a:[a[0]*0.8,a[1]*0.8], env.action_space)

    return env
