import numpy as np

### Data Structure:
# allbounds = [
#     {'N': 2, 'lb': [..], 'ub': [..], 'count': xx},
#     {'N': 3, 'lb': [..], 'ub': [..], 'count': xx},
#     ...
# ]

def get_unconditionalN_bounds(bounds, dim_X):
    total_count = sum([d['count'] for d in bounds])
    lb_sum = np.zeros(dim_X)
    ub_sum = np.zeros(dim_X)
    for d in bounds:
        count_d = d['count']
        lb_d = np.array(d['lb'])
        ub_d = np.array(d['ub'])
        prob_d = count_d/total_count
        lb_sum = np.sum([lb_sum, lb_d*prob_d], axis=0)
        ub_sum = np.sum([ub_sum, ub_d*prob_d], axis=0)
    return lb_sum, ub_sum

