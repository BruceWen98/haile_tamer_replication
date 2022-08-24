import numpy as np

## ---------- Supporting Functions ---------- ##

def gen_values(n, distribution='lognormal'):
    if distribution=='uniform':
        return np.random.uniform(0, 10, n)
    elif distribution=='lognormal':
        return np.random.lognormal(3,1,n)

def new_bid(lbda, bidder_value, increment, curr_bid):
    jump_bool = np.random.binomial(1,lbda)
    # No Jump
    if jump_bool==0:
        # Check that after increment bid <= value
        if curr_bid+increment<=bidder_value:
            return curr_bid+increment
        else:
            return -1
    # Jump 
    elif jump_bool==1:
        # Check that after increment bid <= value
        if curr_bid+increment<=bidder_value:
            return np.random.uniform(curr_bid+increment, bidder_value)
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

def run_1_simulation(n, lbda, increment, distribution='lognormal'):
    # Generate values
    values = list(gen_values(n, distribution))
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
    
    return values, all_bids, player_1, max_bid_dict  ## player_1 is the winning player.



print(run_1_simulation(6,0.1,0.5))
# print(two_bidder_battle(9.20,9.1,0,0.5,0.4))