import numpy as np
import cv2


THETA = 0.5
GRAVITATIONAL_CONSTANT = 1
PHYSICS_SUBSTEPS = 1000  # Number of physics steps per render frame
FRICTION = 0 # not being used
IMAGE_BOTTOM_LEFT = (0, 0)  # (x, y) coordinates of bottom-left corner
IMAGE_TOP_RIGHT = (400, 400)  # (x, y) coordinates of top-right corner
IMAGE_SIZE = 600
IMAGE_BORDER_SIZE = 150

TIME_DELTA = 0.0001


# Particle data structure
point_type = np.dtype([
    ('m', np.float64),
    ('x', np.float64),
    ('y', np.float64),
    ('x_prev', np.float64),
    ('y_prev', np.float64),
    ('fx', np.float64),
    ('fy', np.float64)
])

POINTS = np.array([
    (1e3, 100, 100, 100, 100, 0, 0),
    (1e3, 300, 100, 300, 100, 0, 0),
    (1e3, 200, 200, 200, 200, 0, 0),
], dtype=point_type)


node_type = np.dtype([
    # Physics properties
    ('m', np.float64),              # Mass (or sum of masses for internal nodes)
    ('x', np.float64),              # Center of mass X
    ('y', np.float64),              # Center of mass Y

    # Node type: -1 = internal node, >= 0 = leaf containing particle at this index
    ('particle_idx', np.int32),

    # Spatial bounds
    ('x_min', np.float64),
    ('x_mid', np.float64),
    ('x_max', np.float64),
    ('y_min', np.float64),
    ('y_mid', np.float64),
    ('y_max', np.float64),

    # Children: explicit indices into NODES array [NW, NE, SW, SE], -1 = no child
    ('children', np.int32, (4,)),
])

TREE_SIZE = 0
MAX_TREE_SIZE = 10**4
NODES = np.zeros(MAX_TREE_SIZE, dtype=node_type)


def allocate_node(x_min, x_max, y_min, y_max):
    """Allocate a new tree node and return its index"""
    global TREE_SIZE

    if TREE_SIZE >= MAX_TREE_SIZE:
        raise RuntimeError(f"Tree exceeded max size {MAX_TREE_SIZE}")

    idx = TREE_SIZE
    TREE_SIZE += 1

    # Initialize node
    NODES[idx]['m'] = 0.0
    NODES[idx]['x'] = 0.0
    NODES[idx]['y'] = 0.0
    NODES[idx]['particle_idx'] = -1  # Empty internal node
    NODES[idx]['x_min'] = x_min
    NODES[idx]['x_mid'] = (x_min + x_max) / 2
    NODES[idx]['x_max'] = x_max
    NODES[idx]['y_min'] = y_min
    NODES[idx]['y_mid'] = (y_min + y_max) / 2
    NODES[idx]['y_max'] = y_max
    NODES[idx]['children'][:] = -1  # No children yet

    return idx


def get_quadrant(point, node):
    """Determine which quadrant (0-3) a point belongs to in a node"""
    if point['x'] <= node['x_mid'] and point['y'] <= node['y_mid']:
        return 0  # NW
    elif point['x'] > node['x_mid'] and point['y'] <= node['y_mid']:
        return 1  # NE
    elif point['x'] <= node['x_mid'] and point['y'] > node['y_mid']:
        return 2  # SW
    else:
        return 3  # SE


def get_quadrant_bounds(node, quadrant):
    """Get spatial bounds for a quadrant of a node"""
    if quadrant == 0:  # NW
        return node['x_min'], node['x_mid'], node['y_min'], node['y_mid']
    elif quadrant == 1:  # NE
        return node['x_mid'], node['x_max'], node['y_min'], node['y_mid']
    elif quadrant == 2:  # SW
        return node['x_min'], node['x_mid'], node['y_mid'], node['y_max']
    else:  # SE (quadrant == 3)
        return node['x_mid'], node['x_max'], node['y_mid'], node['y_max']


def add_particle(particle_idx, node_idx):
    """Add a particle to the tree at the given node (pointer-based approach)"""
    global TREE_SIZE

    point = POINTS[particle_idx]
    node = NODES[node_idx]

    # Determine which quadrant this particle belongs to
    quadrant = get_quadrant(point, node)

    # Case 1: Node is empty - make it a leaf
    if node['particle_idx'] == -1 and node['children'][0] == -1:
        NODES[node_idx]['particle_idx'] = particle_idx
        NODES[node_idx]['m'] = point['m']
        NODES[node_idx]['x'] = point['x']
        NODES[node_idx]['y'] = point['y']
        return

    # Case 2: Node is a leaf with existing particle - subdivide!
    if node['particle_idx'] >= 0:
        old_particle_idx = node['particle_idx']
        old_particle = POINTS[old_particle_idx]

        # Convert to internal node
        NODES[node_idx]['particle_idx'] = -1

        # Determine which quadrant the OLD particle belongs to
        old_quadrant = get_quadrant(old_particle, node)

        # Create child for old particle if needed
        if node['children'][old_quadrant] == -1:
            x_min, x_max, y_min, y_max = get_quadrant_bounds(node, old_quadrant)
            child_idx = allocate_node(x_min, x_max, y_min, y_max)
            NODES[node_idx]['children'][old_quadrant] = child_idx

        # Add old particle to its child
        add_particle(old_particle_idx, node['children'][old_quadrant])

    # Case 3: Node is internal - create child if needed and recurse
    # Re-read node after potential modification
    node = NODES[node_idx]

    if node['particle_idx'] == -1:  # Internal node
        child_idx = node['children'][quadrant]

        if child_idx == -1:
            # Create new child for this quadrant
            x_min, x_max, y_min, y_max = get_quadrant_bounds(node, quadrant)
            child_idx = allocate_node(x_min, x_max, y_min, y_max)
            NODES[node_idx]['children'][quadrant] = child_idx

        # Recurse to child with current particle
        add_particle(particle_idx, child_idx)

        # Update center of mass for this node
        update_center_of_mass(node_idx)


def update_center_of_mass(node_idx):
    """Update mass and center of mass from children"""
    node = NODES[node_idx]

    total_m = 0.0
    weighted_x = 0.0
    weighted_y = 0.0

    for child_idx in node['children']:
        if child_idx >= 0:  # Child exists
            child = NODES[child_idx]
            total_m += child['m']
            weighted_x += child['x'] * child['m']
            weighted_y += child['y'] * child['m']

    if total_m > 0:
        NODES[node_idx]['m'] = total_m
        NODES[node_idx]['x'] = weighted_x / total_m
        NODES[node_idx]['y'] = weighted_y / total_m


def reset_tree():
    """Reset tree for new frame"""
    global TREE_SIZE
    TREE_SIZE = 0


def build_tree():
    """Build Barnes-Hut tree from current particle positions"""
    reset_tree()

    # Calculate bounds
    x_min = POINTS['x'].min()
    x_max = POINTS['x'].max()
    y_min = POINTS['y'].min()
    y_max = POINTS['y'].max()

    # Add small padding to avoid particles exactly on boundaries
    padding = 1.0
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    # Create root
    root_idx = allocate_node(x_min, x_max, y_min, y_max)

    # Add all particles
    for i in range(len(POINTS)):
        add_particle(i, root_idx)

    return root_idx
  

# def step_verlet(index: int): # parallelize with this
#     p = POINTS[index]
#     tmp_x, tmp_y = p["x"], p["y"]
#     p["x"] = 2 * p["x"] - p["x_prev"] + p["fx"] * TIME_DELTA ** 2 / p["m"]
#     p["y"] = 2 * p["y"] - p["y_prev"] + p["fy"] * TIME_DELTA ** 2 / p["m"]
#     p["x_prev"], p["y_prev"] = tmp_x, tmp_y
#     p["fx"], p["fy"] = 0, 0

def step_verlet_vectorized():
    """Update all particles at once using NumPy vectorization"""
    tmp_x = POINTS['x'].copy()
    tmp_y = POINTS['y'].copy()
    
    POINTS['x'] = 2 * POINTS['x'] - POINTS['x_prev'] + POINTS['fx'] * TIME_DELTA**2 / POINTS['m']
    POINTS['y'] = 2 * POINTS['y'] - POINTS['y_prev'] + POINTS['fy'] * TIME_DELTA**2 / POINTS['m']
    
    POINTS['x_prev'] = tmp_x
    POINTS['y_prev'] = tmp_y


def calculate_force(particle_idx, node_idx):
    """Calculate force on particle from tree node (pointer-based Barnes-Hut)"""
    if node_idx < 0:
        return  # No node

    point = POINTS[particle_idx]
    node = NODES[node_idx]

    if node['particle_idx'] == particle_idx:
        return

    if node['particle_idx'] >= 0:
        apply_force_particle_to_particle(particle_idx, node['particle_idx'])
        return

    # Internal node - check s/d
    dx = node['x'] - point['x']
    dy = node['y'] - point['y']
    d = (dx**2 + dy**2)**0.5
    s = node['x_max'] - node['x_min']

    if d > 0 and s / d < THETA:
        # Far enough - use center of mass approximation
        apply_force_from_com(particle_idx, node_idx)
    else:
        # Too close - recurse to children
        for child_idx in node['children']:
            if child_idx >= 0:
                calculate_force(particle_idx, child_idx)


def apply_force_particle_to_particle(p1_idx, p2_idx):
    """Apply force between two particles"""
    if p1_idx == p2_idx:
        return

    p1 = POINTS[p1_idx]
    p2 = POINTS[p2_idx]

    dx = p2['x'] - p1['x']
    dy = p2['y'] - p1['y']
    r_squared = max(dx**2 + dy**2, 1e-13)

    F = GRAVITATIONAL_CONSTANT * p1['m'] * p2['m'] / r_squared
    theta = np.arctan2(dy, dx)

    POINTS[p1_idx]['fx'] += F * np.cos(theta)
    POINTS[p1_idx]['fy'] += F * np.sin(theta)


def apply_force_from_com(particle_idx, node_idx):
    """Apply force from node's center of mass to particle"""
    node = NODES[node_idx]
    point = POINTS[particle_idx]

    dx = node['x'] - point['x']
    dy = node['y'] - point['y']
    r_squared = max(dx**2 + dy**2, 1e-13)

    F = GRAVITATIONAL_CONSTANT * point['m'] * node['m'] / r_squared
    theta = np.arctan2(dy, dx)

    POINTS[particle_idx]['fx'] += F * np.cos(theta)
    POINTS[particle_idx]['fy'] += F * np.sin(theta)


def render_points(img: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float):
    """Render particles from NumPy array"""
    # Use the larger of: particle bounds or fixed image bounds
    x_min_final = min(x_min, IMAGE_BOTTOM_LEFT[0])
    x_max_final = max(x_max, IMAGE_TOP_RIGHT[0])
    y_min_final = min(y_min, IMAGE_BOTTOM_LEFT[1])
    y_max_final = max(y_max, IMAGE_TOP_RIGHT[1])

    x_range = x_max_final - x_min_final
    y_range = y_max_final - y_min_final

    # Draw bounding box
    cv2.rectangle(img, (IMAGE_BORDER_SIZE-20, IMAGE_BORDER_SIZE-20),
                 (IMAGE_BORDER_SIZE+IMAGE_SIZE+20, IMAGE_BORDER_SIZE+IMAGE_SIZE+20),
                 (255, 255, 255), 1)

    # Render each particle
    for particle in POINTS:
        # Transform to screen coordinates
        screen_x = int((particle['x'] - x_min_final) * IMAGE_SIZE / x_range)
        screen_y = int((y_max_final - particle['y']) * IMAGE_SIZE / y_range)  # Flip y-axis

        if 0 <= screen_x <= IMAGE_SIZE and 0 <= screen_y <= IMAGE_SIZE:
            px = screen_x + IMAGE_BORDER_SIZE
            py = screen_y + IMAGE_BORDER_SIZE

            # Draw particle
            cv2.circle(img, (px, py), 2, (255, 255, 255), -1)

            # Draw force vector
            force_magnitude = np.sqrt(particle['fx']**2 + particle['fy']**2)
            if force_magnitude > 0:
                scale = 5 * np.log10(force_magnitude + 1) / force_magnitude
                fx_scaled = int(particle['fx'] * scale)
                fy_scaled = -int(particle['fy'] * scale)  # Flip y-axis for force arrow
            else:
                fx_scaled = fy_scaled = 0

            cv2.arrowedLine(img, (px, py), (px + fx_scaled, py + fy_scaled),
                          (255, 255, 255), 1)

            # Draw particle info
            FONT_SIZE = 0.4
            (textWidth, textHeight), baseline = cv2.getTextSize(
                f"x: {round(particle['x'], 4)}, y: {round(particle['y'], 4)}",
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, 1)
            TEXT_LEFT = px + 10
            TEXT_TOP = py - 2*textHeight + 4

            cv2.putText(img, f"p: {round(particle['x'], 2)}, {round(particle['y'], 2)}",
                       (TEXT_LEFT, TEXT_TOP), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 255, 255), 1)
            cv2.putText(img, f"f: {round(particle['fx'], 2)}, {round(particle['fy'], 2)}",
                       (TEXT_LEFT, TEXT_TOP+2*textHeight), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 255, 255), 1)

    return img


def draw_scale_indicators(img: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float):
    """Draw axis tick marks and labels around the bounding box"""
    num_ticks = 5
    tick_length = 10

    # Use the larger of: particle bounds or fixed image bounds
    x_min_final = min(x_min, IMAGE_BOTTOM_LEFT[0])
    x_max_final = max(x_max, IMAGE_TOP_RIGHT[0])
    y_min_final = min(y_min, IMAGE_BOTTOM_LEFT[1])
    y_max_final = max(y_max, IMAGE_TOP_RIGHT[1])

    x_range = x_max_final - x_min_final
    y_range = y_max_final - y_min_final

    # X-axis ticks (bottom)
    for i in range(num_ticks + 1):
        x_val = x_min_final + x_range * i / num_ticks
        x_pixel = IMAGE_BORDER_SIZE + int(IMAGE_SIZE * i / num_ticks)

        # Draw tick mark
        cv2.line(img, (x_pixel, IMAGE_BORDER_SIZE+IMAGE_SIZE+20),
                 (x_pixel, IMAGE_BORDER_SIZE+IMAGE_SIZE+20+tick_length),
                 (255, 255, 255), 1)

        # Draw label
        label = f"{round(x_val, 1)}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        cv2.putText(img, label, (x_pixel - text_size[0]//2, IMAGE_BORDER_SIZE+IMAGE_SIZE+45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Y-axis ticks (left)
    for i in range(num_ticks + 1):
        y_val = y_min_final + y_range * i / num_ticks
        y_pixel = IMAGE_BORDER_SIZE + IMAGE_SIZE - int(IMAGE_SIZE * i / num_ticks)

        # Draw tick mark
        cv2.line(img, (IMAGE_BORDER_SIZE-20-tick_length, y_pixel),
                 (IMAGE_BORDER_SIZE-20, y_pixel),
                 (255, 255, 255), 1)

        # Draw label
        label = f"{round(y_val, 1)}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        cv2.putText(img, label, (IMAGE_BORDER_SIZE-35-text_size[0], y_pixel+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return img


def main():
    """Main simulation loop using pointer-based Barnes-Hut with NumPy"""
    count = 0

    print(f"Starting simulation with {len(POINTS)} particles")
    print(f"THETA = {THETA}, G = {GRAVITATIONAL_CONSTANT}, dt = {TIME_DELTA}")
    print(f"Physics substeps per frame: {PHYSICS_SUBSTEPS}")

    while True:
        for _ in range(PHYSICS_SUBSTEPS):
            root_idx = build_tree()
            
            POINTS['fx'] = 0
            POINTS['fy'] = 0
            
            for i in range(len(POINTS)): # hard to vectorize
                calculate_force(i, root_idx)
            
            step_verlet_vectorized()

        img = np.zeros((IMAGE_SIZE + 2*IMAGE_BORDER_SIZE, IMAGE_SIZE + 2*IMAGE_BORDER_SIZE))

        # Get rendering bounds
        x_min = POINTS['x'].min()
        x_max = POINTS['x'].max()
        y_min = POINTS['y'].min()
        y_max = POINTS['y'].max()

        img = render_points(img, x_min, x_max, y_min, y_max)
        img = draw_scale_indicators(img, x_min, x_max, y_min, y_max)

        # Frame counter
        cv2.putText(img, f"Frame: {str(count).zfill(5)} ({PHYSICS_SUBSTEPS} steps/frame)",
                   (IMAGE_BORDER_SIZE-20, IMAGE_BORDER_SIZE-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
        cv2.putText(img, f"Tree nodes: {TREE_SIZE}",
                   (IMAGE_BORDER_SIZE-20, IMAGE_BORDER_SIZE-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

        cv2.imshow('Particles', img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        count += 1

    print(f'\nSimulation ended')
    print(f'Total frames: {count}')
    print(f'Total physics steps: {count * PHYSICS_SUBSTEPS}')
    print('\nFinal particle states:')
    for i, p in enumerate(POINTS):
        print(f"  Particle {i}: pos=({p['x']:.2f}, {p['y']:.2f}), "
              f"vel=({(p['x']-p['x_prev'])/TIME_DELTA:.2f}, {(p['y']-p['y_prev'])/TIME_DELTA:.2f}), "
              f"m={p['m']:.0f}")


if __name__ == '__main__':
    main()