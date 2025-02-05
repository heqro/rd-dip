import numpy as np
import matplotlib.pyplot as plt
from torch.special import i0, i1
import torch


def r(u, f, sigma=0.15):
    t = u * f / (sigma**2)
    num = i1(t)
    den = i0(t)
    return num / den


sigma = 0.15
u = torch.linspace(-1, 1, 1000)
th = np.sqrt(2 * sigma**2)
print(f"Threshold: {th}")

f_positive = th + 0.35  # f > sqrt(2 * sigma**2)
f_negative = th - 0.35  # f < sqrt(2 * sigma**2)

# Calculations for f = th + 0.35
org_pos = u**2 / (2 * sigma**2) - torch.log(i0(u * f_positive / sigma**2))
functional_pos = (u - r(u, f_positive) * f_positive) ** 2
functional2_pos = (u / r(u, f_positive) - f_positive) ** 2

# Calculations for f = th - 0.35
org_neg = u**2 / (2 * sigma**2) - torch.log(i0(u * f_negative / sigma**2))
functional_neg = (u - r(u, f_negative) * f_negative) ** 2
functional2_neg = (u / r(u, f_negative) - f_negative) ** 2

# Plot for f = th + 0.35
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.plot(u, functional_pos, label="$(u - rf)^2$")
plt.plot(u, functional2_pos, label="$(u/r - f)^2$")
plt.title(r"$f = \sqrt{2}\sigma  + 0.35$")
plt.xlabel("$u$")
plt.grid()
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(u, org_pos, label="Rician fidelity")
plt.title(r"$f = \sqrt{2}\sigma + 0.35$")
plt.xlabel("$u$")
plt.grid()
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(u, u / r(u, f_positive), label="Quotient u/r")
plt.title(r"$f = \sqrt{2}\sigma + 0.35$")
plt.xlabel("$u$")
plt.grid()
plt.legend()

# Plot for f = th - 0.35
plt.subplot(2, 3, 4)
plt.plot(u, functional_neg, label="$(u - rf)^2$")
plt.plot(u, functional2_neg, label="$(u/r - f)^2$")
plt.title(r"$f = \sqrt{2}\sigma - 0.35$")
plt.xlabel("$u$")
plt.grid()
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(u, org_neg, label="Rician fidelity")
plt.title(r"$f = \sqrt{2}\sigma - 0.35$")
plt.xlabel("$u$")
plt.grid()
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(u, u / r(u, f_negative), label="Quotient u/r")
plt.title(r"$f = \sqrt{2}\sigma - 0.35$")
plt.xlabel("$u$")
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig("rician_subplots.pdf", bbox_inches="tight")
plt.close()
