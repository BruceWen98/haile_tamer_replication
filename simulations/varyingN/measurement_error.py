import numpy as np

for k in range(2,50):
    for n in range(2,10):
        for m in range(n+1,11):
            right = (n+k) / (m-1+k) / (m+k)
            left = n/(m-1)/m
            if left >= right:
                pass
            else:
                print(k,n,m,left,right)