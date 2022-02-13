# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
params = np.linspace(0, 1, 101)
likelihoods = params ** 6 * (1 - params) ** 4


# %%
plt.xlabel("θ")
plt.ylabel("Likelihood")
plt.title("Likelihood vs θ (Dataset 1)")
plt.scatter(params, likelihoods)
plt.show()

# %%
print(max(likelihoods))
print(np.argmax(likelihoods))
# %%

# %%
likelihoods2 = params ** 3 * (1 - params) ** 2

likelihoods3 = params ** 60 * (1 - params) ** 40

likelihoods4 = params ** 5 * (1 - params) ** 5
# %%
plt.xlabel("θ")
plt.ylabel("Likelihood")
plt.title("Likelihood vs θ (Dataset 1)")
plt.scatter(params, likelihoods)
plt.show()

plt.xlabel("θ")
plt.ylabel("Likelihood")
plt.title("Likelihood vs θ (Dataset 2)")
plt.scatter(params, likelihoods2)
plt.show()

plt.xlabel("θ")
plt.ylabel("Likelihood")
plt.title("Likelihood vs θ (Dataset 3)")
plt.scatter(params, likelihoods3)
plt.show()

plt.xlabel("θ")
plt.ylabel("Likelihood")
plt.title("Likelihood vs θ (Dataset 4)")
plt.scatter(params, likelihoods4)
plt.show()


# %%
