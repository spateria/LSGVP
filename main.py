import copy
import tensorflow as tf
import numpy as np
import tqdm

from agent import UvfAgent
from train_agent import train_eval
from LSGVP import LSGVP_Model
from envs.twoD_mazes import env_load_fn
from test_goalseeking import run_goalseeking

######################################## Define Args ###########################################
max_episode_steps = 20
replay_buffer_size = 1000 
edge_fraction = 0.15
edge_distance_cutoff = 1

envss = ['FourRooms', 'Maze6x6', 'Maze11x11']
env_name = envss[1] # Choose one of the environments shown above. 
resize_factor = 5  # Inflate the environment to increase the difficulty.


########################################## Train UVF Agent #######################################

tf.compat.v1.reset_default_graph()

tf_env = env_load_fn(env_name, max_episode_steps,
                     resize_factor=resize_factor,
                     terminate_on_timeout=False)
eval_tf_env = env_load_fn(env_name, max_episode_steps,
                          resize_factor=resize_factor,
                          terminate_on_timeout=True)


############################# Sample random states into replay buffer. State space is bounded, so we can assume the agent has already seen all the states during training ################
eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
    prob_constraint=0.0,
    min_dist=0,
    max_dist=np.inf)
rb_vec = []

for _ in tqdm.tnrange(replay_buffer_size):
  ts = eval_tf_env.reset()
  rb_vec.append(ts.observation['observation'].numpy()[0])
rb_vec = np.array(rb_vec)


########################### Generate subgoal graph model ###########################################
lsgvp = LSGVP_Model()
lsgvp.env = eval_tf_env
lsgvp.actor = agent

lsgvp.state_repo = copy.deepcopy(rb_vec)
subgoals = lsgvp.graph_abstraction()
lsgvp.edge_distance_cutoff = edge_distance_cutoff

if subgoals is not None:
    lsgvp.create_subgoal_graph(subgoals, edge_fraction)

original_graph = copy.deepcopy(lsgvp.subgoal_graph)
pruned_graph = lsgvp.prune_graph(original_graph)

lsgvp.display_spatial_graph(original_graph, lsgvp.node_to_subgoal)
lsgvp.display_spatial_graph(pruned_graph, lsgvp.node_to_subgoal)
print(len(list(pruned_graph.nodes)))
print(len(list(pruned_graph.edges)))

############################# Use the pruned subgoal graph for long-horizon planning to diverse goals #########################
run_goalseeking(eval_tf_env, agent, lsgvp)
