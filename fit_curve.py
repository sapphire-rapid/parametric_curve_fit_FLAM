"""
Parametric curve parameter estimation
--------------------------------------

Given:
    x(t) = t*cos(theta) - e^(M|t|) * sin(0.3t) * sin(theta) + X
    y(t) = 42 + t*sin(theta) + e^(M|t|) * sin(0.3t) * cos(theta)

    unknowns: theta (0-50 deg), M (-0.05 to 0.05), X (0-100)
    parameter: t in [6, 60]

We are given 1500 (x, y) points ("xy_data.csv") that lie on this curve for
various (unordered / unknown) values of t in [6, 60], and must recover
theta, M, X.

Run:
    pip install -r requirements.txt
    python fit_curve.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution, minimize

DATA_PATH = "xy_data.csv"
T_MIN, T_MAX = 6.0, 60.0


def curve_xy(theta_deg, M, X, t):
    """Evaluate the parametric curve at parameter value(s) t."""
    theta = np.radians(theta_deg)
    v = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * np.cos(theta) - v * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + v * np.cos(theta)
    return np.column_stack([x, y])


def pca_initial_theta(pts):
    """
    Structural insight: (x - X, y - 42) = R(theta) @ (t, e^(M|t|) sin(0.3t)).
    Since t dominates the second coordinate over most of [6, 60], the point
    cloud's dominant principal axis approximates the direction of the curve
    at angle theta (mod 180 degrees). This gives a fast, cheap initial guess
    to sanity check / seed the global optimizer.
    """
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(principal[1], principal[0])) % 180
    return angle


def chamfer_loss(params, pts, t_dense):
    """
    Mean L1 distance from each data point to the nearest point on the
    candidate curve. We use this "nearest point on curve" (Chamfer-style)
    loss instead of ordinary index-matched least squares because the rows
    in xy_data.csv are NOT ordered by t -- there is no known correspondence
    between a given (x, y) row and a specific t value.
    """
    theta, M, X = params
    curve_pts = curve_xy(theta, M, X, t_dense)
    tree = cKDTree(curve_pts)
    d, _ = tree.query(pts, p=1)
    return d.mean()


def fit(pts):
    t_dense = np.linspace(T_MIN, T_MAX, 6000)
    bounds = [(0, 50), (-0.05, 0.05), (0, 100)]

    # Stage 1: global search (bounded, handles the sin(0.3t) wiggle without
    # getting stuck in a bad local minimum).
    de_result = differential_evolution(
        chamfer_loss, bounds, args=(pts, t_dense),
        tol=1e-10, seed=1, maxiter=200, popsize=25, polish=True,
    )

    # Stage 2: local polish for extra precision.
    refined = minimize(
        chamfer_loss, de_result.x, args=(pts, t_dense),
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 3000},
    )
    return refined.x, refined.fun


def main():
    df = pd.read_csv(DATA_PATH)
    pts = df[["x", "y"]].values

    pca_angle = pca_initial_theta(pts)
    print(f"[sanity check] PCA-estimated theta (mod 180 deg): {pca_angle:.2f} deg")

    (theta, M, X), loss = fit(pts)
    print("\nFitted parameters:")
    print(f"  theta = {theta:.4f} deg  ({np.radians(theta):.4f} rad)")
    print(f"  M     = {M:.4f}")
    print(f"  X     = {X:.4f}")
    print(f"  mean L1 residual to nearest curve point: {loss:.5f}")

    # Verification plot
    t_dense = np.linspace(T_MIN, T_MAX, 3000)
    curve_pts = curve_xy(theta, M, X, t_dense)

    plt.figure(figsize=(8, 6))
    plt.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.4, color="tab:orange", label="given data")
    plt.plot(curve_pts[:, 0], curve_pts[:, 1], color="tab:blue", linewidth=2,
              label=f"fitted curve (theta={theta:.2f} deg, M={M:.4f}, X={X:.2f})")
    plt.legend()
    plt.title("Fitted parametric curve vs given data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.savefig("assets/fit_check.png", dpi=130, bbox_inches="tight")
    print("\nSaved verification plot to assets/fit_check.png")


if __name__ == "__main__":
    main()
