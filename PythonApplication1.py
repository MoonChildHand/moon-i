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
    """Calculate the electromagnetic force: F_em = rho_e * E + J × B."""
    # Compute current density J = ∇ × B (z-direction for 2D)
    J_x = partial_derivative(B[:, :, 1], dx, axis=1) - partial_derivative(B[:, :, 0], dy, axis=0)
    J = np.stack([np.zeros_like(J_x), np.zeros_like(J_x), J_x], axis=2)  # 2D curl of B in z-direction

    # Lorentz force: F_em = rho_e * E + J × B
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
    나비에-스토크스 방정식에서 비선형 항을 제거하고 전자기력으로 대체한 선형화된 형태.
    선형 항 (시간 변화, 압력 구배, 점성 항)은 그대로 유지.
    """
    for _ in range(nt):
        un = u.copy()
        vn = v.copy()
        pn = p.copy()

        # 전류 밀도 계산: J = ∇ × B (z 방향으로만 존재, 2D의 경우)
        J_x = partial_derivative(B[:, :, 1], dx, axis=1) - partial_derivative(B[:, :, 0], dy, axis=0)
        J = np.stack([np.zeros_like(J_x), np.zeros_like(J_x), J_x], axis=2)  # 전류 밀도 벡터

        # 전자기력 계산: F_em = rho_e * E + J × B
        F_em = rho_e[:, :, None] * E + np.cross(J, B, axis=2)

        # x 방향 속도 업데이트 (선형 항만 포함)
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            + dt * (
                - (1 / rho) * partial_derivative(pn, dx, axis=1)[1:-1, 1:-1]  # 압력 구배
                + mu * laplacian(un, dx, dy)[1:-1, 1:-1]                     # 점성 항
                + F_em[1:-1, 1:-1, 0] / rho                                 # 전자기력 x 성분
            )
        )

        # y 방향 속도 업데이트
        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            + dt * (
                - (1 / rho) * partial_derivative(pn, dy, axis=0)[1:-1, 1:-1]  # 압력 구배
                + mu * laplacian(vn, dx, dy)[1:-1, 1:-1]                     # 점성 항
                + F_em[1:-1, 1:-1, 1] / rho                                 # 전자기력 y 성분
            )
        )

        # 압력 보정 (연속 방정식 기반)
        p[1:-1, 1:-1] = (
            pn[1:-1, 1:-1]
            - dt * rho * (
                partial_derivative(un, dx, axis=1)[1:-1, 1:-1]
                + partial_derivative(vn, dy, axis=0)[1:-1, 1:-1]
            )
        )

        # 경계 조건 적용
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
