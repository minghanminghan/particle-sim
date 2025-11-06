import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from barnes_hut import (
    POINT_TYPE, NODE_TYPE, TIME_DELTA, GRAVITATIONAL_CONSTANT,
    MAX_TREE_SIZE, build_tree, calculate_force, step_verlet_vectorized, THETA
)


def three_body_derivatives(t, y):
    """
    Compute derivatives for 3-body problem.
    y = [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
    """
    positions = y[:6].reshape(3, 2)
    velocities = y[6:].reshape(3, 2)

    # Masses (all equal to 1.0)
    masses = np.array([1.0, 1.0, 1.0])

    # Compute accelerations
    accelerations = np.zeros((3, 2))

    for i in range(3):
        for j in range(3):
            if i != j:
                r_vec = positions[j] - positions[i]
                r_squared = max(np.sum(r_vec**2), 1e-13)
                r = np.sqrt(r_squared)

                # F = G * m1 * m2 / r^2
                # a = F / m1 = G * m2 / r^2
                force_magnitude = GRAVITATIONAL_CONSTANT * masses[j] / r_squared
                accelerations[i] += force_magnitude * (r_vec / r)

    # Return derivatives: [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]
    derivatives = np.concatenate([velocities.flatten(), accelerations.flatten()])
    return derivatives


def run_ground_truth_simulation(initial_conditions, t_end, dt_output):
    """
    Run high-precision simulation using scipy.

    Args:
        initial_conditions: [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
        t_end: End time
        dt_output: Time step for output

    Returns:
        times, positions, velocities
    """
    t_eval = np.arange(0, t_end, dt_output)

    sol = solve_ivp(
        three_body_derivatives,
        (0, t_end),
        initial_conditions,
        method='DOP853',  # High-order Runge-Kutta
        t_eval=t_eval,
        rtol=1e-12,
        atol=1e-15
    )

    positions = sol.y[:6, :].T.reshape(-1, 3, 2)  # (time, particle, xy)
    velocities = sol.y[6:, :].T.reshape(-1, 3, 2)

    return sol.t, positions, velocities


def run_verlet_simulation(initial_positions, initial_velocities, n_steps, dt, store_every=10000):
    """
    Run Verlet integration using Barnes-Hut tree structure.

    Args:
        initial_positions: (3, 2) array
        initial_velocities: (3, 2) array
        n_steps: Number of time steps
        dt: Time step
        store_every: Store results every N steps (default 1000)

    Returns:
        positions, velocities (arrays of shape (n_stored, 3, 2))
    """
    # Calculate how many snapshots we'll store
    n_stored = (n_steps - 1) // store_every + 2
    positions = np.zeros((n_stored, 3, 2))
    velocities = np.zeros((n_stored, 3, 2))

    # Create POINTS array in the same format as barnes_hut.py
    points = np.zeros(3, dtype=POINT_TYPE)

    for i in range(3):
        points[i]['m'] = 1.0
        points[i]['x'] = initial_positions[i, 0]
        points[i]['y'] = initial_positions[i, 1]
        points[i]['x_prev'] = initial_positions[i, 0] - initial_velocities[i, 0] * dt
        points[i]['y_prev'] = initial_positions[i, 1] - initial_velocities[i, 1] * dt
        points[i]['fx'] = 0.0
        points[i]['fy'] = 0.0

    # Store initial state
    positions[0] = initial_positions
    velocities[0] = initial_velocities

    # Allocate tree nodes
    nodes = np.empty(MAX_TREE_SIZE, dtype=NODE_TYPE)

    stored_idx = 1
    for step in range(1, n_steps):
        if step % 100000 == 0:
            print(f"Step {step:,}/{n_steps:,} ({100*step/n_steps:.1f}%)")

        # Build Barnes-Hut tree
        root_idx, tree_size = build_tree(points, nodes)

        # Reset forces
        points['fx'] = 0
        points['fy'] = 0

        # Calculate forces using Barnes-Hut
        for i in range(len(points)):
            calculate_force(i, root_idx, points, nodes, THETA)

        # Update positions using Verlet integration
        step_verlet_vectorized(points)

        # Store results only at specified intervals
        if step % store_every == 0 or step == n_steps - 1:
            positions[stored_idx, :, 0] = points['x']
            positions[stored_idx, :, 1] = points['y']
            velocities[stored_idx, :, 0] = (points['x'] - points['x_prev']) / dt
            velocities[stored_idx, :, 1] = (points['y'] - points['y_prev']) / dt
            stored_idx += 1

    return positions[:stored_idx], velocities[:stored_idx]


def compare_integrators():
    """Compare Verlet vs ground truth."""

    # Periodic 3-body initial conditions (from barnes_hut.py)
    # Particle 1: position (-1, 0), velocity (0.3471168881, 0.5327249454)
    # Particle 2: position (1, 0), velocity (0.3471168881, 0.5327249454)
    # Particle 3: position (0, 0), velocity (-0.6942337762, -1.0654498908)
    initial_positions = np.array([
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0]
    ])

    initial_velocities = np.array([
        [0.3471168881, 0.5327249454],
        [0.3471168881, 0.5327249454],
        [-0.6942337762, -1.0654498908]
    ])

    # Simulation parameters
    dt = TIME_DELTA          # time step
    t_end = 1                # seconds
    n_steps = int(t_end / dt)
    store_every = 1000       # Store every N steps (reduces memory/time by 1000x)

    # Run ground truth
    print("Running ground truth simulation...")
    initial_conditions = np.concatenate([
        initial_positions.flatten(),
        initial_velocities.flatten()
    ])

    times_gt, positions_gt, velocities_gt = run_ground_truth_simulation(
        initial_conditions, t_end, dt * store_every  # Match output sampling rate
    )

    # Run Verlet
    print("Running Verlet simulation...")
    positions_verlet, velocities_verlet = run_verlet_simulation(
        initial_positions, initial_velocities, n_steps, dt, store_every
    )

    # Ensure arrays have the same length (ground truth may have one extra point)
    min_len = min(len(positions_verlet), len(positions_gt))
    positions_verlet = positions_verlet[:min_len]
    velocities_verlet = velocities_verlet[:min_len]
    positions_gt = positions_gt[:min_len]
    velocities_gt = velocities_gt[:min_len]
    times_gt = times_gt[:min_len]

    print(f"Comparing {min_len} time steps...")

    # Compute errors
    print("\nComputing errors...")
    position_errors = np.linalg.norm(positions_verlet - positions_gt, axis=(1, 2))
    velocity_errors = np.linalg.norm(velocities_verlet - velocities_gt, axis=(1, 2))

    # Compute energy (for conservation check)
    def compute_energy(pos, vel):
        """Compute total energy."""
        # Kinetic energy
        KE = 0.5 * np.sum(vel**2, axis=(1, 2))

        # Potential energy
        PE = np.zeros(len(pos))
        for i in range(3):
            for j in range(i+1, 3):
                r = np.linalg.norm(pos[:, i] - pos[:, j], axis=1)
                PE -= GRAVITATIONAL_CONSTANT * 1.0 * 1.0 / r

        return KE + PE

    energy_gt = compute_energy(positions_gt, velocities_gt)
    energy_verlet = compute_energy(positions_verlet, velocities_verlet)
    energy_error = np.abs(energy_verlet - energy_gt)

    # Print statistics
    print("\n=== Error Statistics ===")
    print(f"Position error (mean): {np.mean(position_errors):.2e}")
    print(f"Position error (max): {np.max(position_errors):.2e}")
    print(f"Velocity error (mean): {np.mean(velocity_errors):.2e}")
    print(f"Velocity error (max): {np.max(velocity_errors):.2e}")
    print(f"Energy error (mean): {np.mean(energy_error):.2e}")
    print(f"Energy error (max): {np.max(energy_error):.2e}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Position error over time
    axes[0, 0].semilogy(times_gt, position_errors)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position Error (L2 norm)')
    axes[0, 0].set_title('Position Error vs Time')
    axes[0, 0].grid(True)

    # Velocity error over time
    axes[0, 1].semilogy(times_gt, velocity_errors)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity Error (L2 norm)')
    axes[0, 1].set_title('Velocity Error vs Time')
    axes[0, 1].grid(True)

    # Energy error over time
    axes[1, 0].semilogy(times_gt, energy_error)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Energy Error')
    axes[1, 0].set_title('Energy Conservation Error vs Time')
    axes[1, 0].grid(True)

    # Trajectories comparison (all particles)
    colors = ['red', 'green', 'blue']
    for i in range(3):
        axes[1, 1].plot(positions_gt[:, i, 0], positions_gt[:, i, 1],
                       color=colors[i], linestyle='-', linewidth=2,
                       label=f'GT P{i}', alpha=0.7)
        axes[1, 1].plot(positions_verlet[:, i, 0], positions_verlet[:, i, 1],
                       color=colors[i], linestyle='--', linewidth=1.5,
                       label=f'Verlet P{i}', alpha=0.7)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title('All Particle Trajectories (Solid=GT, Dashed=Verlet)')
    axes[1, 1].legend(fontsize=8, ncol=2)
    axes[1, 1].grid(True)
    axes[1, 1].axis('equal')

    plt.tight_layout()
    # plt.savefig('test/verlet_accuracy_test.png', dpi=150)
    # print("\nPlot saved as 'verlet_accuracy_test.png'")
    plt.show()


if __name__ == '__main__':
    compare_integrators()
