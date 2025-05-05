# regressao-multivariada-no-norm.py
"""
@file regressao-multivariada-no-norm.py
@brief Multivariate linear regression exercise with gradient descent and normal equation.
@details Este script executa um fluxo de trabalho completo para regressão linear multivariada,
          incluindo normalização de features, cálculo de parâmetros via gradiente descendente
          e equação normal, além de comparação de custos.
@author Diogo Brasil Da Silva <diogobrasildasilva@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from RegressionMultivariate.compute_cost_multi import compute_cost_multi
from RegressionMultivariate.gradient_descent_multi import gradient_descent_multi
from RegressionMultivariate.gradient_descent_multi import gradient_descent_multi_with_history
from RegressionMultivariate.normal_eqn import normal_eqn

def costs_from_history(X_b: np.ndarray, y: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    return np.array([compute_cost_multi(X_b, y, th) for th in thetas])

def main():
    os.makedirs("Figures/no_norm/", exist_ok=True)

    data = np.loadtxt('Data/ex1data2.txt', delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    m = len(y)

    print('Primeiros 10 exemplos de treinamento:')
    print(np.column_stack((X[:10], y[:10])))

    X_b =  np.column_stack((np.ones((m, 1)), X))

    alpha = 2e-9
    num_iters = 4000
    theta_gd = np.zeros(X_b.shape[1])
    theta_gd, J_history = gradient_descent_multi(X_b, y, theta_gd, alpha, num_iters)
    print('\nTheta via Gradient Descent:')
    print(theta_gd)

    plt.figure()
    plt.plot(np.arange(1, num_iters + 1), J_history, 'b-', linewidth=2)
    plt.xlabel('Iteração')
    plt.ylabel('Custo J(\u03b8)')
    plt.title('Convergência do Gradiente (Sem Normalização)')
    plt.grid(True)
    plt.savefig('Figures/no_norm/convergencia_custo_multi.png', dpi=300)
    plt.show()

    example = np.array([1, 1650, 3])
    price_gd = example @ theta_gd
    print(f'\nPreço previsto (GD) para [1650,3]: ${price_gd:.2f}')

    X_ne = X_b.copy()
    theta_ne = normal_eqn(X_ne, y)
    price_ne = example @ theta_ne
    print('\nTheta via Equação Normal:')
    print(theta_ne)
    print(f'Preço previsto (NE) para [1650,3]: ${price_ne:.2f}')

    cost_ne = compute_cost_multi(X_ne, y, theta_ne)
    print(f'\n[CUSTO NE] Custo usando θ_ne em X (original): {cost_ne:.2f}')

    plt.figure()
    plt.plot(np.arange(1, num_iters + 1), J_history, 'b-', label='Gradiente Descendente')
    plt.hlines(cost_ne, 1, num_iters, colors='r', linestyles='--', label='Equação Normal')
    plt.xlabel('Iteração')
    plt.ylabel('Custo J(\u03b8)')
    plt.title('GD vs Equação Normal (Sem Normalização)')
    plt.legend()
    plt.grid(True)
    plt.savefig('Figures/no_norm/convergencia_custo_vs_ne.png', dpi=300)
    plt.show()

    theta_gd, J_history, theta_history = gradient_descent_multi_with_history(X_b, y, np.zeros(X_b.shape[1]), alpha, num_iters)

    t1_hist, t2_hist = theta_history[:, 1], theta_history[:, 2]
    max_dev1 = np.max(np.abs(t1_hist - theta_ne[1]))
    max_dev2 = np.max(np.abs(t2_hist - theta_ne[2]))
    span1 = span2 = 1.5 * max(max_dev1, max_dev2)

    t1_vals = np.linspace(theta_ne[1] - span1, theta_ne[1] + span1, 120)
    t2_vals = np.linspace(theta_ne[2] - span2, theta_ne[2] + span2, 120)
    T1, T2 = np.meshgrid(t1_vals, t2_vals)

    J_mesh = np.zeros_like(T1)
    for i in range(T1.shape[0]):
        for j in range(T1.shape[1]):
            J_mesh[i, j] = compute_cost_multi(X_b, y, [theta_ne[0], T1[i, j], T2[i, j]])

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T1, T2, J_mesh, cmap="viridis", alpha=0.85, linewidth=0)
    ax.plot(t1_hist, t2_hist, costs_from_history(X_b, y, theta_history), "r.-", label="Trajetória GD")
    ax.scatter(theta_ne[1], theta_ne[2], compute_cost_multi(X_b, y, theta_ne),
               s=80, marker="x", color="black", linewidths=2, label="NE")
    fig.colorbar(surf, ax=ax, shrink=0.6, label="Custo J(\u03b8)")
    ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$"); ax.set_zlabel("Custo J(\u03b8)")
    ax.set_title("Superfície J(\u03b81, \u03b82) (Sem Normalização)")
    ax.view_init(elev=30, azim=-60)
    ax.legend()
    fig.savefig("Figures/no_norm/superficie_GD_vs_NE.png", dpi=300)

    from matplotlib.colors import LogNorm
    plt.figure(figsize=(7, 5))
    levels = np.logspace(np.log10(J_mesh.min()), np.log10(J_mesh.max()), 60)
    cf = plt.contourf(T1, T2, J_mesh, levels=levels, norm=LogNorm(), cmap="viridis")
    plt.colorbar(cf, label="Custo J(\u03b8)")
    plt.plot(t1_hist, t2_hist, "r.-", ms=2, label="Trajetória GD")
    plt.scatter(theta_ne[1], theta_ne[2], s=80, marker="x", color="black", label="NE")
    plt.xlabel(r"$\theta_1$"); plt.ylabel(r"$\theta_2$")
    plt.title("Contorno J(\u03b81, \u03b82) (Sem Normalização)"); plt.legend()
    plt.savefig("Figures/no_norm/contorno_GD_vs_NE.png", dpi=300)
    plt.show()

    fig2 = plt.figure(figsize=(7, 5))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.scatter(X[:, 0], X[:, 1], y, c="red", marker="x", label="Dados de treino")

    f1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 40)
    f2_vals = np.linspace(X[:, 1].min(), X[:, 1].max(), 40)
    F1, F2 = np.meshgrid(f1_vals, f2_vals)
    Z = theta_gd[0] + theta_gd[1] * F1 + theta_gd[2] * F2

    surf2 = ax2.plot_surface(
        F1, F2, Z, alpha=0.5, cmap="viridis", rstride=1, cstride=1
    )

    ax2.set_xlabel("Tamanho (pés²)")
    ax2.set_ylabel("Quartos")
    ax2.set_zlabel("Preço (US$)")
    ax2.set_title("Ajuste da Regressão Linear Multivariada (Sem Normalização)")
    ax2.view_init(elev=25, azim=-135)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([], [], color="red", marker="x", linestyle="", label="Dados de treino"),
        Patch(facecolor=surf2.get_facecolor()[0], edgecolor="none", alpha=0.5, label="Plano GD"),
    ]
    ax2.legend(handles=handles)
    fig2.tight_layout()
    fig2.savefig("Figures/no_norm/ajuste_regressao_multivariada.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
