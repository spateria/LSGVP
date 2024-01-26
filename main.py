import copy
import tensorflow as tf
import numpy as np
import tqdm

from LSGVP import LSGVP_Model
from test_goalseeking import run_goalseeking

######################################## Define Args ###########################################
replay_buffer_size = 1000 
edge_fraction = 0.15
edge_distance_cutoff = 1

########### Load agent and environment #############
agent = None
eval_tf_env = None

## Sample random states into replay buffer. State space is bounded, so we can assume the agent has already seen all the states during training ##
eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
    prob_constraint=0.0,
    min_dist=0,
    max_dist=np.inf)
rb_vec = []

for _ in tqdm.tnrange(replay_buffer_size):
  ts = eval_tf_env.reset()
  rb_vec.append(ts.observation['observation'].numpy()[0])
rb_vec = np.array(rb_vec) #this stores randomly sampled states from which subgoals are chosen for graph building


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
