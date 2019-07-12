
import numpy as np
import itertools
import pypownet.agent
import pypownet.environment
import os
import pypownet.environment

input_dir = 'public_data'

def set_environement(game_level = "datasets", start_id=0):
    """
        Load the first chronic (scenario) in the directory public_data/datasets
    """
    return pypownet.environment.RunEnv(parameters_folder=os.path.abspath(input_dir),
                                              game_level=game_level,
                                              chronic_looping_mode='natural', start_id=start_id,
                                              game_over_mode="hard")




env1 = set_environement()
env1.reset()
action_space = env1.action_space
observation_space = env1.observation_space
game = env1.game

valid_sw=dict()
valid_actions_list=[]
# Num of elements per SE
print ('Numbers of elements of sub-station:')
print ('++  ++  ++  ++  ++  ++  ++  ++  ++')
action_index=0
col_sub_index=[]
row_action_index=[]
for i in env1.action_space.substations_ids:
    print ('SE: {:d} - # elements: {}'.format(int(i), env1.action_space.get_number_elements_of_substation(i)))
    swtiching_patterns= ["".join(seq) for seq in itertools.product("01", repeat=env1.action_space.get_number_elements_of_substation(i)-1)] # reduce by 1 bit due to compliment being the same
    swtiching_patterns = [ [int(sw_i_k) for sw_i_k in '0'+sw_i] for sw_i in swtiching_patterns ] # adding back the '0' at the beginning as we are fizing this bit
    for sw_action in swtiching_patterns:
        # Initialize action class
        applied_action = action_space.get_do_nothing_action(as_class_Action=True)
        #  Set new switches to the new state.
        applied_action.set_substation_switches(i, sw_action)
        valid_actions_list.append(np.ndarray.tolist(applied_action.as_array()));
        row_action_index.append(action_index)
        col_sub_index.append(int(i) - 1)
        action_index += 1
        # Simulate one step in the environment
        obs, *_ = env1.simulate(applied_action)
        try :
            len(obs)
            obs = observation_space.array_to_observation(obs)
            if (sum(obs.are_productions_cut)==0 and sum(obs.are_loads_cut)==0):
                valid_sw.setdefault(int(i),[]).append(sw_action)
        except :
            0

from scipy.sparse import coo_matrix, spdiags

valid_actions_masking_subid=coo_matrix((np.ones(len(col_sub_index)),(row_action_index,col_sub_index)),shape=(len(col_sub_index),14))
valid_actions_array=np.array(valid_actions_list)

valid_actions_array_uniq ,uniq_index = np.unique(valid_actions_array, return_index=True, axis=0)
valid_actions_masking_subid_perm = valid_actions_masking_subid.toarray()[uniq_index,:]
valid_actions_masking_subid_perm[0,0]=0 # this makes sure that the no-action is never prohibited
np.savez_compressed('valid_actions_array_uniq.npz', valid_actions_array_uniq=valid_actions_array_uniq)
print("Determined the unique substation switching actions and saved in .npz files")

np.savez_compressed('valid_actions_masking_subid_perm.npz', valid_actions_masking_subid_perm=valid_actions_masking_subid_perm)
print("Determined the mapping between substation-ID and switching actions and saved in .npz files")

