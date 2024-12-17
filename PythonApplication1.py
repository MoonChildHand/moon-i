import numpy as np

# === Helper Functions ===
def partial_derivative(f, dx, axis):
    """Calculate partial derivative using central difference."""
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * dx)

def laplacian(f, dx, dy):
    """Calculate 2D Laplacian."""
    d2f_dx2 = (np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)) / dx**2
    d2f_dy2 = (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / dy**2
    return d2f_dx2 + d2f_dy2

def calculate_electromagnetic_force(rho_e, E, B, dx, dy):
    """Calculate the electromagnetic force: F_em = rho_e * E + J �� B."""
    # Compute current density J = �� �� B (z-direction for 2D)
    J_x = partial_derivative(B[:, :, 1], dx, axis=1) - partial_derivative(B[:, :, 0], dy, axis=0)
    J = np.stack([np.zeros_like(J_x), np.zeros_like(J_x), J_x], axis=2)  # 2D curl of B in z-direction

    # Lorentz force: F_em = rho_e * E + J �� B
    F_em = rho_e[:, :, None] * E + np.cross(J, B, axis=2)
    return F_em

# === Navier-Stokes Solver with Electromagnetic Forces ===
def navier_stokes_with_em(u, v, p, rho, mu, dx, dy, dt, nt, rho_e, E, B):
    for _ in range(nt):
        un = u.copy()
        vn = v.copy()
        pn = p.copy()

        # Calculate electromagnetic force
        F_em = calculate_electromagnetic_force(rho_e, E, B, dx, dy)

        # Update x-direction velocity
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            + dt * (
                - (1 / rho) * partial_derivative(pn, dx, axis=1)[1:-1, 1:-1]  # Pressure gradient
                + mu * laplacian(un, dx, dy)[1:-1, 1:-1]                     # Viscous term
                + F_em[1:-1, 1:-1, 0] / rho                                 # EM force (x-component)
            )
        )

        # Update y-direction velocity
        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            + dt * (
                - (1 / rho) * partial_derivative(pn, dy, axis=0)[1:-1, 1:-1]  # Pressure gradient
                + mu * laplacian(vn, dx, dy)[1:-1, 1:-1]                     # Viscous term
                + F_em[1:-1, 1:-1, 1] / rho                                 # EM force (y-component)
            )
        )

        # Pressure correction
        p[1:-1, 1:-1] = pn[1:-1, 1:-1] - dt * rho * (
            partial_derivative(un, dx, axis=1)[1:-1, 1:-1]
            + partial_derivative(vn, dy, axis=0def navier_stokes_linear_with_em(u, v, p, rho, mu, dx, dy, dt, nt, rho_e, E, B):
    """
    ����-����ũ�� �����Ŀ��� ���� ���� �����ϰ� ���ڱ������ ��ü�� ����ȭ�� ����.
    ���� �� (�ð� ��ȭ, �з� ����, ���� ��)�� �״�� ����.
    """
    for _ in range(nt):
        un = u.copy()
        vn = v.copy()
        pn = p.copy()

        # ���� �е� ���: J = �� �� B (z �������θ� ����, 2D�� ���)
        J_x = partial_derivative(B[:, :, 1], dx, axis=1) - partial_derivative(B[:, :, 0], dy, axis=0)
        J = np.stack([np.zeros_like(J_x), np.zeros_like(J_x), J_x], axis=2)  # ���� �е� ����

        # ���ڱ�� ���: F_em = rho_e * E + J �� B
        F_em = rho_e[:, :, None] * E + np.cross(J, B, axis=2)

        # x ���� �ӵ� ������Ʈ (���� �׸� ����)
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            + dt * (
                - (1 / rho) * partial_derivative(pn, dx, axis=1)[1:-1, 1:-1]  # �з� ����
                + mu * laplacian(un, dx, dy)[1:-1, 1:-1]                     # ���� ��
                + F_em[1:-1, 1:-1, 0] / rho                                 # ���ڱ�� x ����
            )
        )

        # y ���� �ӵ� ������Ʈ
        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            + dt * (
                - (1 / rho) * partial_derivative(pn, dy, axis=0)[1:-1, 1:-1]  # �з� ����
                + mu * laplacian(vn, dx, dy)[1:-1, 1:-1]                     # ���� ��
                + F_em[1:-1, 1:-1, 1] / rho                                 # ���ڱ�� y ����
            )
        )

        # �з� ���� (���� ������ ���)
        p[1:-1, 1:-1] = (
            pn[1:-1, 1:-1]
            - dt * rho * (
                partial_derivative(un, dx, axis=1)[1:-1, 1:-1]
                + partial_derivative(vn, dy, axis=0)[1:-1, 1:-1]
            )
        )

        # ��� ���� ����
        u[:, 0] = u[:, -1] = u[0, :] = u[-1, :] = 0
        v[:, 0] = v[:, -1] = v[0, :] = v[-1, :] = 0
        p[:, 0] = p[:, -1] = p[0, :] = p[-1, :] = 0

    return u, v, p
)[1:-1, 1:-1]
        )

        # Apply boundary conditions
        u[:, 0] = u[:, -1] = u[0, :] = u[-1, :] = 0
        v[:, 0] = v[:, -1] = v[0, :] = v[-1, :] = 0
        p[:, 0] = p[:, -1] = p[0, :] = p[-1, :] = 0

    return u, v, p
