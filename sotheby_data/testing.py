import pickle


with open('output_dicts_jump2.pickle', 'rb') as handle:
    output_dicts_jump2 = pickle.load(handle)

with open('output_dicts_NOjump2.pickle', 'rb') as handle:
    output_dicts_NOjump2 = pickle.load(handle)

print(output_dicts_jump2[:3])
print("\n\n")
print(output_dicts_NOjump2[:3])