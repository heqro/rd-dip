import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# Define symbols
u, f, sigma = sp.symbols("u f sigma", real=True, positive=True)

# Define R function using modified Bessel functions
i0 = sp.besseli(0, u * f / sigma**2)
i1 = sp.besseli(1, u * f / sigma**2)
r = i1 / i0  # type: ignore

# Define threshold
sigma_val = 0.2
th = sp.sqrt(2 * sigma_val**2)

bump = 0.19
# Values for f
f_positive = th + bump  # type: ignore
f_negative = th - bump  # type: ignore

# Functional forms
org_expr = u**2 / (2 * sigma**2) - sp.log(i0)
rician_norm_bad = (u - r * f) ** 2
rician_norm_good = (u / r - f) ** 2
quotient_expr = u / r
rician_norm_bad_diff1 = sp.diff(rician_norm_bad, u)
rician_norm_bad_diff2 = sp.diff(rician_norm_bad_diff1, u)
rician_norm_good_diff1 = sp.diff(rician_norm_good, u)
rician_norm_good_diff2 = sp.diff(rician_norm_good_diff1, u)

# Simbolic to numeric conversion
org_pos_fun = sp.lambdify(u, org_expr.subs({sigma: sigma_val, f: f_positive}))
rn_bad_pos = sp.lambdify(u, rician_norm_bad.subs({sigma: sigma_val, f: f_positive}))
rn_good_pos = sp.lambdify(u, rician_norm_good.subs({sigma: sigma_val, f: f_positive}))
quotient_pos = sp.lambdify(u, quotient_expr.subs({sigma: sigma_val, f: f_positive}))
rn_bad_diff1_pos = sp.lambdify(
    u, rician_norm_bad_diff1.subs({sigma: sigma_val, f: f_positive})
)
rn_good_diff1_pos = sp.lambdify(
    u, rician_norm_good_diff1.subs({sigma: sigma_val, f: f_positive})
)
rn_bad_diff2_pos = sp.lambdify(
    u, rician_norm_bad_diff2.subs({sigma: sigma_val, f: f_positive})
)
rn_good_diff2_pos = sp.lambdify(
    u, rician_norm_good_diff2.subs({sigma: sigma_val, f: f_positive})
)

org_neg_fun = sp.lambdify(u, org_expr.subs({sigma: sigma_val, f: f_negative}))
rn_bad_neg = sp.lambdify(u, rician_norm_bad.subs({sigma: sigma_val, f: f_negative}))
rn_good_neg = sp.lambdify(u, rician_norm_good.subs({sigma: sigma_val, f: f_negative}))
quotient_neg = sp.lambdify(u, quotient_expr.subs({sigma: sigma_val, f: f_negative}))
rn_bad_diff1_neg = sp.lambdify(
    u, rician_norm_bad_diff1.subs({sigma: sigma_val, f: f_negative})
)
rn_good_diff1_neg = sp.lambdify(
    u, rician_norm_good_diff1.subs({sigma: sigma_val, f: f_negative})
)
rn_bad_diff2_neg = sp.lambdify(
    u, rician_norm_bad_diff2.subs({sigma: sigma_val, f: f_negative})
)
rn_good_diff2_neg = sp.lambdify(
    u, rician_norm_good_diff2.subs({sigma: sigma_val, f: f_negative})
)

# Numeric evaluation
u_vals = np.linspace(-1, 1, 1000)


n_cols = 4


def full_plot():
    plt.figure(figsize=(18, 6))

    # Upwards bump
    ax1 = plt.subplot(2, n_cols, 1)
    ax1.plot(u_vals, rn_bad_pos(u_vals), label="$(u - rf)^2$")
    ax1.plot(u_vals, rn_good_pos(u_vals), label="$(u/r - f)^2$")
    ax1.grid(True, which="both")  # Ensure both vertical and horizontal grid lines

    ax2 = ax1.twinx()
    ax2.plot(
        u_vals,
        org_pos_fun(u_vals),
        label="Rician fidelity",
        color="green",
        linestyle="dashed",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")  # Single legend
    plt.xlabel("$u$")
    plt.title(r"$f = \sqrt{2}\sigma+" + f"{bump}$")

    plt.subplot(2, n_cols, 2)
    plt.plot(u_vals, quotient_pos(u_vals), label="Quotient u/r")
    plt.title(r"$f = \sqrt{2}\sigma+" + f"{bump}$")
    plt.xlabel("$u$")
    plt.grid()
    plt.legend()

    plt.subplot(2, n_cols, 3)
    plt.plot(u_vals, rn_bad_diff1_pos(u_vals), label=r"$d(u-rf)^2/du$")
    plt.plot(u_vals, rn_good_diff1_pos(u_vals), label=r"$d(u/r-f)^2/du$")
    plt.title(r"$f = \sqrt{2}\sigma+" + f"{bump}$")
    plt.xlabel("$u$")
    plt.grid()
    plt.legend()

    plt.subplot(2, n_cols, 4)
    plt.plot(u_vals, rn_bad_diff2_pos(u_vals), label=r"$d^2(u-rf)^2/du^2$")
    plt.plot(u_vals, rn_good_diff2_pos(u_vals), label=r"$d^2(u/r-f)^2/du^2$")
    plt.title(r"$f = \sqrt{2}\sigma+" + f"{bump}$")
    plt.xlabel("$u$")
    plt.grid()
    plt.legend()

    # Downwards bump (?)
    ax1 = plt.subplot(2, n_cols, 5)
    ax1.plot(u_vals, rn_bad_neg(u_vals), label="$(u - rf)^2$")
    ax1.plot(u_vals, rn_good_neg(u_vals), label="$(u/r - f)^2$")
    ax1.grid(True, which="both")
    ax2 = ax1.twinx()
    ax2.plot(
        u_vals,
        org_neg_fun(u_vals),
        label="Rician fidelity",
        color="green",
        linestyle="dashed",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")  # Single legend
    plt.title(r"$f = \sqrt{2}\sigma-" + f"{bump}$")
    plt.xlabel("$u$")

    plt.subplot(2, n_cols, 6)
    plt.plot(u_vals, quotient_neg(u_vals), label="Quotient u/r")
    plt.title(r"$f = \sqrt{2}\sigma-" + f"{bump}$")
    plt.xlabel("$u$")
    plt.grid()
    plt.legend()

    plt.subplot(2, n_cols, 7)
    plt.plot(u_vals, rn_bad_diff1_neg(u_vals), label=r"$d(u-rf)^2/du$")
    plt.plot(u_vals, rn_good_diff1_neg(u_vals), label=r"$d(u/r-f)^2/du$")
    plt.title(r"$f = \sqrt{2}\sigma-" + f"{bump}$")
    plt.xlabel("$u$")
    plt.grid()
    plt.legend()

    plt.subplot(2, n_cols, 8)
    plt.plot(u_vals, rn_bad_diff2_neg(u_vals), label=r"$d^2(u-rf)^2/du^2$")
    plt.plot(u_vals, rn_good_diff2_neg(u_vals), label=r"$d^2(u/r-f)^2/du^2$")
    plt.title(r"$f = \sqrt{2}\sigma-" + f"{bump}$")
    plt.xlabel("$u$")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig("rician_subplots_sympy.pdf", bbox_inches="tight")
    plt.close()


def small_plot():
    plt.figure(figsize=(8, 4))

    # Upwards bump
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(u_vals, rn_bad_pos(u_vals), label="$(u - rf)^2$")
    ax1.plot(u_vals, rn_good_pos(u_vals), label="$(u/r - f)^2$")
    ax1.grid(True, which="both")  # Ensure both vertical and horizontal grid lines

    ax2 = ax1.twinx()
    ax2.plot(
        u_vals,
        org_pos_fun(u_vals),
        label="Rician fidelity",
        color="green",
        linestyle="dashed",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")  # Single legend
    ax1.set_xlabel("$u$")
    plt.title(r"$f = \sqrt{2}\sigma+" + f"{bump}$")

    ax1 = plt.subplot(1, 2, 2)
    ax1.plot(u_vals, rn_bad_neg(u_vals), label="$(u - rf)^2$")
    ax1.plot(u_vals, rn_good_neg(u_vals), label="$(u/r - f)^2$")
    ax1.grid(True, which="both")
    ax2 = ax1.twinx()
    ax2.plot(
        u_vals,
        org_neg_fun(u_vals),
        label="Rician fidelity",
        color="green",
        linestyle="dashed",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")  # Single legend
    plt.title(r"$f = \sqrt{2}\sigma-" + f"{bump}$")
    ax1.set_xlabel("$u$")

    plt.tight_layout()
    plt.savefig("rician_subplots_sympy.pdf", bbox_inches="tight")
    plt.close()


def med_plot():
    from matplotlib.lines import Line2D

    plt.figure(figsize=(8, 4))

    # Upwards bump
    ax = plt.subplot(2, 2, 1)
    ax.plot(u_vals, rn_bad_pos(u_vals), label="$(u - rf)^2$", color="blue")
    ax.plot(u_vals, rn_good_pos(u_vals), label="$(u/r - f)^2$", color="orange")
    ax.grid(True)  # Ensure both vertical and horizontal grid lines
    ax.set_xlabel("$u$")
    ax.set_title(r"$f^2 > 2\sigma^2$")

    ax = plt.subplot(2, 2, 2)
    ax.plot(u_vals, rn_bad_neg(u_vals), label="$(u - rf)^2$", color="blue")
    ax.plot(u_vals, rn_good_neg(u_vals), label="$(u/r - f)^2$", color="orange")
    ax.grid(True)
    ax.set_xlabel("$u$")
    plt.title(r"$f^2 < 2\sigma^2$")

    ax = plt.subplot(2, 2, 3)
    ax.plot(
        u_vals,
        org_pos_fun(u_vals),
        label="Rician fidelity",
        color="green",
    )
    ax.grid(True)
    ax.set_xlabel("$u$")

    ax = plt.subplot(2, 2, 4)
    ax.plot(
        u_vals,
        org_neg_fun(u_vals),
        label="Rician fidelity",
        color="green",
    )
    ax.grid(True)
    ax.set_xlabel("$u$")

    legend_elements = [
        Line2D([0], [0], color="blue", lw=2, label=r"$(u - rf)^2$"),
        Line2D([0], [0], color="orange", lw=2, label=r"$(u/r - f)^2$"),
        Line2D([0], [0], color="green", lw=2, label="Rician fidelity"),
    ]
    plt.figlegend(
        handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07)
    )

    plt.tight_layout()
    plt.savefig("rician_subplots_sympy.pdf", bbox_inches="tight")
    plt.close()


full_plot()
