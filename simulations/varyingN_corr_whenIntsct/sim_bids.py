import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

## ---------- Supporting Functions ---------- ##

def gen_values(n, distribution='lognormal'):
    num_low = sum(np.random.choice(2,n))
    vals_high = np.random.lognormal(2.5, 0.5, n-num_low)
    vals_low = np.random.lognormal(2.0, 0.5, num_low)
    all_vals = np.concatenate((vals_high, vals_low))
    return all_vals

def generate_correlated_lognormal(n, mu, sigma, rho):
    '''
    n: number of values to generate
    mu: mean of the logarithm of the values
    sigma: standard deviation of the logarithm of the values
    rho: correlation coefficient
    '''
    # Generate covariance matrix
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                cov_matrix[i,j] = sigma**2
            else:
                cov_matrix[i,j] = sigma * sigma * rho
    
    # Generate multivariate lognormal distribution
    mean = np.ones(n) * mu
    correlated_lognormal = np.random.multivariate_normal(mean, cov_matrix)
    return np.exp(correlated_lognormal)

def new_bid(lbda, bidder_value, increment, curr_bid):
    jump_bool = np.random.binomial(1,lbda)
    # No Jump
    if jump_bool==0:
        # Check that after increment bid <= value
        if curr_bid*(1+increment)<=bidder_value:
            return curr_bid*(1+increment)
        else:
            return -1
    # Jump 
    elif jump_bool==1:
        # Check that after increment bid <= value
        if curr_bid*(1+increment)<=bidder_value:
            return np.random.uniform(curr_bid*(1+increment), bidder_value)
        else:
            return -1

def pick_2_players(player_list): 
    # Pick 2 different players at random, from list of remaining players.
    player_1,player_2 = np.random.choice(player_list, 2, replace=False)
    return player_1, player_2

def two_bidder_battle(player1_v, player2_v, curr_bid, increment, lbda):
    # Check that after increment bid <= value
    first_to_bid = np.random.choice([1,2], 1, p=[0.5,0.5])[0]
    p1_bids = []
    p2_bids = []
    all_bids = []
    losing_player = 0

    if first_to_bid == 1:
        bid_old1 = curr_bid
        bid_new1 = -1 #initialize this as -1, out of the distribution support. 
        while True:
            bid_new1 = new_bid(lbda, player1_v, increment, bid_old1)
            if bid_new1 == -1:
                losing_player = 1
                break
            
            bid_new2 = new_bid(lbda, player2_v, increment, bid_new1)
            if bid_new2 == -1:
                p1_bids.append(bid_new1)
                all_bids.append(bid_new1)
                losing_player = 2
                break

            bid_old1 = bid_new2
            p1_bids.append(bid_new1)
            p2_bids.append(bid_new2)
            all_bids.append(bid_new1)
            all_bids.append(bid_new2)
            
    elif first_to_bid == 2:
        bid_old2 = curr_bid
        bid_new2 = -1 #initialize this as -1, out of the distribution support. 
        while True:
            bid_new2 = new_bid(lbda, player2_v, increment, bid_old2)
            if bid_new2 == -1:
                losing_player = 2
                break

            bid_new1 = new_bid(lbda, player1_v, increment, bid_new2)
            if bid_new1 == -1:
                p2_bids.append(bid_new2)
                all_bids.append(bid_new2)
                losing_player = 1
                break

            bid_old2 = bid_new1
            p1_bids.append(bid_new1)
            p2_bids.append(bid_new2)
            all_bids.append(bid_new1)
            all_bids.append(bid_new2)


    try: 
        p1_max_bid = max(p1_bids)
    except ValueError:
        p1_max_bid = 0
    try: 
        p2_max_bid = max(p2_bids)
    except ValueError:
        p2_max_bid = 0
    all_bids = list(sorted(all_bids))
    return all_bids, p1_max_bid, p2_max_bid, losing_player




## ---------- Simulation Function ---------- ##

def run_1_simulation(n, lbda, increment, distribution='lognormal', corr=False, rho_corr = 0.5):
    # Generate values
    if corr==False:
        values = list(gen_values(n, distribution))
    else:
        values = list(generate_correlated_lognormal(n, 2.5, 0.5, rho_corr))
    # Initialize current bid
    curr_bid = 0
    # Initialize list of bids
    all_bids = []
    # Initialize dictionary of max bids corresponding to each player.
    max_bid_dict = {}

    # Initialize list of remaining players. Initially just range(len(values)).
    # A player is identified by her index in the list of values.
    remaining_players = list(range(len(values)))

    ## Run the simulation
    player_1, player_2 = pick_2_players(remaining_players)
    while len(remaining_players)>1:
        # 1. Pick the new player to replace the losing previous player 2.
        player_2 = np.random.choice(list(filter(lambda x: x!= player_1, remaining_players)), 1)[0]
        player_1v = values[player_1]
        player_2v = values[player_2]

        # 2. Battle between 2 players.
        all_bids_1battle, p1_max_bid, p2_max_bid, losing_player = two_bidder_battle(player_1v, player_2v, curr_bid, increment, lbda)
        
        # 3. Update remaining players, update all bids.
        if losing_player==1:
            remaining_players.remove(player_1)
            max_bid_dict[player_1] = p1_max_bid
            player_1 = player_2  ## We will assign player_1 as the new player each round.
        elif losing_player==2:
            remaining_players.remove(player_2)
            max_bid_dict[player_2] = p2_max_bid
        
        if len(all_bids_1battle)>=1:
            all_bids.extend(all_bids_1battle)
            curr_bid = all_bids_1battle[-1]

    # Update winning bidder's max bid.
    max_bid_dict[player_1] = max(all_bids)
    
    # return values, all_bids, player_1, max_bid_dict  ## player_1 is the winning player.
    
    # Re-order max bid dict so that winner is 1, second is 2.
    high1 = max(max_bid_dict.values())
    high2 = max([max_bid_dict[i] for i in max_bid_dict.keys() if max_bid_dict[i]!=high1])
    out_dict = {}
    out_dict['1'] = high1
    out_dict['2'] = high2
    out_dict['n'] = n
    return out_dict

def gen_many_simulations(n, lbda, increment, distribution='lognormal', num_simulations=10000, corr=False, rho_corr=0.5):
    out_list = []
    for i in tqdm(range(num_simulations)):
        out_list.append(run_1_simulation(n, lbda, increment, distribution='lognormal', corr=corr, rho_corr = rho_corr))
    return out_list

def gen_many_simulations_varyingN(lbda, increment, distribution='lognormal', num_simulations=10000, corr=False, rho_corr=0.5):
    out_list = []
    for i in tqdm(range(num_simulations)):
        n = np.random.choice([3,4,5,6,7,8,9,10,11,12], 1)[0]
        out_list.append(run_1_simulation(n, lbda, increment, distribution='lognormal', corr=corr, rho_corr = rho_corr))
    return out_list

# print(run_1_simulation(3,0.1,0.1, corr=True, rho_corr=0.5))

for rho_corr in [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]:
    out_dicts = gen_many_simulations_varyingN(0.01,0.1, num_simulations=10000, corr=True, rho_corr=rho_corr)
    out_dicts = [d for d in out_dicts if d['2']!=0.0]
    print(out_dicts[:100])
    pickle.dump(out_dicts, open('out_dicts_varyingN_corr_{}.p'.format(rho_corr), 'wb'))