![화면 캡처 2024-12-17 102134](https://github.com/user-attachments/assets/58305a3e-b443-4496-b733-183241760be2)
![화면 캡처 2024-12-17 102211](https://github.com/user-attachments/assets/6b74e00f-d10d-4374-8678-d3e1b8c5c02a)
![화면 캡처 2024-12-17 102236](https://github.com/user-attachments/assets/419ed07a-52af-49e2-8324-8702c8332db0)
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
            + partial_derivative(vn, dy, axis=0)[1:-1, 1:-1]
        )

        # Apply boundary conditions
        u[:, 0] = u[:, -1] = u[0, :] = u[-1, :] = 0
        v[:, 0] = v[:, -1] = v[0, :] = v[-1, :] = 0
        p[:, 0] = p[:, -1] = p[0, :] = p[-1, :] = 0

    return u, v, p
![화면 캡처 2024-12-17 102016](https://github.com/user-attachments/assets/d39bf356-a9a2-4810-84f6-473334a5f8de)
