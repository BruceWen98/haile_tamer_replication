import numpy as np
import math
    
def find_nearest_below(arr, x):
    low = 0
    high = len(arr) - 1

    while low < high:
        mid = (low + high + 1) // 2
        if arr[mid] > x:
            high = mid - 1
        else:
            low = mid
    return low

def find_nearest_above(arr, x):
    low = 0
    high = len(arr) - 1

    while low < high:
        mid = (low + high) // 2
        if arr[mid] <= x:
            low = mid + 1
        else:
            high = mid
    if low == len(arr) - 1 and arr[low] <= x:  # x is larger than all elements
        return low
    return high  # otherwise return the index of the element just above x


## TESTING
print(find_nearest_below([0.1,0.15,0.24,0.35,0.53,0.75], 100))
#5

print(find_nearest_below([0.1,0.15,0.24,0.35,0.53,0.75], 0.75))
#5

print(find_nearest_below([0.1,0.15,0.24,0.35,0.53,0.75], 0.53))
# 4

print(find_nearest_below([0.1,0.15,0.24,0.35,0.53,0.75], 0.23))
# 1


print(find_nearest_above([0.1,0.15,0.24,0.35,0.53,0.75], 0.76))
#5

print(find_nearest_above([0.1,0.15,0.24,0.35,0.53,0.75], 0.75))
#5

print(find_nearest_above([0.1,0.15,0.24,0.35,0.53,0.75], 0.36))
# 4

print(find_nearest_above([0.1,0.15,0.24,0.35,0.53,0.75], 0.10001))
# 1

print(find_nearest_above([0.1,0.15,0.24,0.35,0.53,0.75], 0.01))
# 0