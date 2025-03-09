

# Adding module path to sys path if not there, so rl_coach submodules can be imported
import os
import sys
import tensorflow as tf
module_path = os.path.abspath(os.path.join('..'))
resources_path = os.path.abspath(os.path.join('Resources'))
if module_path not in sys.path:
    sys.path.append(module_path)
if resources_path not in sys.path:
    sys.path.append(resources_path)
    
from rl_coach.coach import CoachInterface
from rl_coach.environments.gym_environment import GymEnvironment, GymVectorEnvironment
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import EnvironmentSteps



tf.reset_default_graph()
coach = CoachInterface(preset='CartPole_ClippedPPO')

# registering an iteration signal before starting to run
coach.graph_manager.log_signal('iteration', -1)

coach.graph_manager.heatup(EnvironmentSteps(100))

# training
for it in range(10):
    # logging the iteration signal during training
    coach.graph_manager.log_signal('iteration', it)
    # using the graph manager to train and act a given number of steps
    coach.graph_manager.train_and_act(EnvironmentSteps(100))
    # reading signals during training
    training_reward = coach.graph_manager.get_signal_value('Training Reward')



# inference
env_params = GymVectorEnvironment(level='CartPole-v0')
env = GymEnvironment(**env_params.__dict__, visualization_parameters=VisualizationParameters())

response = env.reset_internal_state()
for _ in range(10):
    action_info = coach.graph_manager.get_agent().choose_action(response.next_state)
    print("State:{}, Action:{}".format(response.next_state,action_info.action))
    response = env.step(action_info.action)
    print("Reward:{}".format(response.reward))




