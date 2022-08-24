test_dict = {0: 8, 1: 14, 2: 0, 3: 19.902419649277878, 4: 0, 5: 20.902419649277878}

def top2(d):
    top2keys = sorted(d, key=d.get, reverse=True)[:2]
    out_dict = {}
    out_dict[top2keys[0]] = d[top2keys[0]]
    out_dict[top2keys[1]] = d[top2keys[1]]
    return out_dict

print(top2(test_dict))