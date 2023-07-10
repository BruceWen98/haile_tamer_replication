# This plot draws and explains the problem when we take the difference in profit bounds.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import seaborn as sns

# Create a vector of x values
x = np.linspace(0, 5, 500)

# Generate a concave CDF similar to a lognormal distribution
s = 0.954  # shape parameter for the lognormal distribution
lognorm_cdf = lognorm.cdf(x, s)

# Generate a convex CDF using a polynomial function
poly_cdf = x ** 3 / max(x) ** 3

# Ensure both CDFs intersect at 0 and highest value
lognorm_cdf = lognorm_cdf / lognorm_cdf[-1]
poly_cdf = poly_cdf / poly_cdf[-1]

# Find the x values where the CDFs equal 0.4
x_val_concave_04 = x[np.abs(lognorm_cdf - 0.4).argmin()]
x_val_convex_04 = x[np.abs(poly_cdf - 0.4).argmin()]
x_val_convex_02 = x[np.abs(poly_cdf - 0.2).argmin()]

# Create the permissible CDF
third_cdf = np.hstack((lognorm_cdf[x <= x_val_concave_04],
                       np.repeat(0.4, np.sum((x > x_val_concave_04) & (x < x_val_convex_04))),
                       poly_cdf[x >= x_val_convex_04]))

# Create the impermissible CDF
fourth_cdf = np.hstack((lognorm_cdf[x <= x_val_concave_04],
                        np.linspace(0.4, 0.2, np.sum((x > x_val_concave_04) & (x < x_val_convex_02))),
                        poly_cdf[x >= x_val_convex_02]))


# Create the plot
sns.set_style('darkgrid')
plt.figure(figsize=(10, 10), tight_layout=True)
plt.plot(x, lognorm_cdf, label='Concave CDF (Lognormal)', linestyle='dashed',linewidth='2', color='tab:blue')
plt.plot(x, poly_cdf, label='Convex CDF (Polynomial)', linestyle='dashed',linewidth='2', color='tab:blue')
plt.plot(x, third_cdf, label='Permissible CDF',linestyle='solid', color='tab:green', alpha=0.6)
plt.plot(x, fourth_cdf, label='Impermissible CDF',linestyle='dashdot', color='tab:red', alpha=0.6)

plt.text(1.5, 0.80, r'$G_{n:n}(v)$', color='tab:blue', fontsize=14)
plt.text(4.1, 0.50, r'$\phi_{n-1:n}(G_{n:n}(v))^n$', color='tab:blue', fontsize=14)
plt.text(1.6, 0.42, r'Permissible $F_{n:n}(v)$', color='tab:green', fontsize=14)
plt.text(0.7, 0.25, r'Impermissible $F_{n:n}(v)$', color='tab:red', fontsize=14)

plt.xticks([x[0],x_val_concave_04,x_val_convex_02,x[-1]], ['0',r'$r_1$',r'$r_2$',r'$\bar{v}$'])
plt.xlabel('Value')
plt.ylabel('CDF')
plt.title(r'Permissible $F_{n:n}(v)$ given its Bounds')

# Vertical lines
plt.axvline(x=x_val_concave_04, ymax=0.405, linewidth=0.7, color='black', linestyle='--')
plt.axvline(x=x_val_convex_02, ymax=0.405, linewidth=0.7, color='black', linestyle='--')

# Plot and write coordinates
points       = [(x_val_concave_04, 0.4), (x_val_convex_02, 0.2), (x_val_convex_02, 0.4)]
point_labels = [r'($r_1$, $0.4$)', r'($r_2$, $0.2$)', r'($r_2$, $0.4$)']
for i,point in enumerate(points):
    plt.scatter(*point, color='black')
    # plt.text(point[0]+0.1, point[1], point_labels[i])

# Explain F_nn(r) must be at least the max(.,.)
# draw arrow
plt.annotate(r'$F_{n:n}(r) \geq \max\{G_{n:n}(r_1), \phi_{n-1:n}(G_{n:n}(r_2))^n \}$', 
             xy=(x_val_convex_02, 0.4), xytext=(x_val_convex_02-0.9, 0.6), arrowprops=dict(arrowstyle="->", color='black'))
    
# Display the plot
plt.savefig('permissible_Fnn.png', dpi=300)


