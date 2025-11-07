import numpy as np
import cv2
from numba import njit


THETA = 0.0
GRAVITATIONAL_CONSTANT = 1
PHYSICS_SUBSTEPS = 10000  # Number of physics steps per render frame
IMAGE_BOTTOM_LEFT = (-3, -3)  # (x, y) coordinates of bottom-left corner
IMAGE_TOP_RIGHT = (3, 3)  # (x, y) coordinates of top-right corner
IMAGE_SIZE = 600
IMAGE_BORDER_SIZE = 150

TIME_DELTA = 1e-7

MAX_TREE_SIZE = 10**4

# Particle data structure
POINT_TYPE = np.dtype([
    ('m', np.float64),
    ('x', np.float64),
    ('y', np.float64),
    ('x_prev', np.float64),   # For Verlet
    ('y_prev', np.float64),   # For Verlet
    ('vx', np.float64),       # For Leapfrog
    ('vy', np.float64),       # For Leapfrog
    ('fx', np.float64),
    ('fy', np.float64)
], align=True)

# Node data structure
NODE_TYPE = np.dtype([
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
], align=True)


@njit
def allocate_node(tree_size: int, nodes, x_min: float, x_max: float, y_min: float, y_max: float):
    """Allocate a new tree node and return (new_tree_size, node_idx)"""
    if tree_size >= MAX_TREE_SIZE:
        raise RuntimeError(f"Tree exceeded max size {MAX_TREE_SIZE}")

    idx = tree_size

    # Initialize node
    nodes[idx]['m'] = 0.0
    nodes[idx]['x'] = 0.0
    nodes[idx]['y'] = 0.0
    nodes[idx]['particle_idx'] = -1  # Empty internal node
    nodes[idx]['x_min'] = x_min
    nodes[idx]['x_mid'] = (x_min + x_max) / 2
    nodes[idx]['x_max'] = x_max
    nodes[idx]['y_min'] = y_min
    nodes[idx]['y_mid'] = (y_min + y_max) / 2
    nodes[idx]['y_max'] = y_max
    nodes[idx]['children'][:] = -1  # No children yet

    return tree_size + 1, idx


@njit
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


@njit
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


@njit
def add_particle(tree_size: int, particle_idx: int, node_idx: int, points, nodes):
    """Add a particle to the tree iteratively. Returns: final tree_size"""
    # Work stack: (particle_idx, node_idx, update_com_flag)
    # update_com_flag indicates if we should update COM for this node after processing
    work_stack = [(particle_idx, node_idx)]
    # Stack to track nodes we need to update COM for
    com_update_stack = []

    while work_stack:
        current_particle_idx, current_node_idx = work_stack.pop()

        point = points[current_particle_idx]
        node = nodes[current_node_idx]

        # Determine which quadrant this particle belongs to
        quadrant = get_quadrant(point, node)

        # Case 1: Node is empty - make it a leaf
        if node['particle_idx'] == -1 and node['children'][0] == -1:
            nodes[current_node_idx]['particle_idx'] = current_particle_idx
            nodes[current_node_idx]['m'] = point['m']
            nodes[current_node_idx]['x'] = point['x']
            nodes[current_node_idx]['y'] = point['y']
            continue  # Done with this particle, process next from stack

        # Case 2: Node is a leaf with existing particle - subdivide!
        elif node['particle_idx'] >= 0:
            old_particle_idx = node['particle_idx']
            old_particle = points[old_particle_idx]

            # Convert to internal node
            nodes[current_node_idx]['particle_idx'] = -1

            # Determine which quadrant the OLD particle belongs to
            old_quadrant = get_quadrant(old_particle, node)

            # Create child for old particle if needed
            if node['children'][old_quadrant] == -1:
                x_min, x_max, y_min, y_max = get_quadrant_bounds(node, old_quadrant)
                tree_size, child_idx = allocate_node(tree_size, nodes, x_min, x_max, y_min, y_max)
                nodes[current_node_idx]['children'][old_quadrant] = child_idx

            # Mark that we need to update COM for current node
            com_update_stack.append(current_node_idx)

            # Add both particles to work stack: old particle in its quadrant, current in its quadrant
            # Process them by adding to the stack (they'll be processed in reverse order)
            old_child_idx = nodes[current_node_idx]['children'][old_quadrant]
            work_stack.append((old_particle_idx, old_child_idx))

            # Now handle the current particle - need to descend to its quadrant
            # Check if child exists for current particle's quadrant
            if node['children'][quadrant] == -1:
                x_min, x_max, y_min, y_max = get_quadrant_bounds(node, quadrant)
                tree_size, child_idx = allocate_node(tree_size, nodes, x_min, x_max, y_min, y_max)
                nodes[current_node_idx]['children'][quadrant] = child_idx

            current_child_idx = nodes[current_node_idx]['children'][quadrant]
            work_stack.append((current_particle_idx, current_child_idx))

        # Case 3: Node is internal - descend to child
        else:  # node['particle_idx'] == -1
            child_idx = node['children'][quadrant]

            if child_idx == -1:
                # Create new child for this quadrant
                x_min, x_max, y_min, y_max = get_quadrant_bounds(node, quadrant)
                tree_size, child_idx = allocate_node(tree_size, nodes, x_min, x_max, y_min, y_max)
                nodes[current_node_idx]['children'][quadrant] = child_idx

            # Mark that we need to update COM for current node
            com_update_stack.append(current_node_idx)

            # Continue with child
            work_stack.append((current_particle_idx, child_idx))

    # Update center of mass for all nodes (in reverse order of traversal)
    for node_idx in com_update_stack[::-1]:
        update_center_of_mass(node_idx, nodes)

    return tree_size


@njit
def update_center_of_mass(node_idx, nodes):
    """Update mass and center of mass from children"""
    node = nodes[node_idx]

    total_m = 0.0
    weighted_x = 0.0
    weighted_y = 0.0

    # TODO remove conditionals
    for child_idx in node['children']:
        if child_idx >= 0:  # Child exists
            child = nodes[child_idx]
            total_m += child['m']
            weighted_x += child['x'] * child['m']
            weighted_y += child['y'] * child['m']

    if total_m > 0:
        nodes[node_idx]['m'] = total_m
        nodes[node_idx]['x'] = weighted_x / total_m
        nodes[node_idx]['y'] = weighted_y / total_m


@njit
def build_tree(points, nodes):
    """Build Barnes-Hut tree from current particle positions. Returns: (root_idx, final_tree_size)"""
    tree_size = 0

    # Calculate bounds
    x_min = points['x'].min()
    x_max = points['x'].max()
    y_min = points['y'].min()
    y_max = points['y'].max()

    # Add small padding to avoid particles exactly on boundaries
    padding = 1.0
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    # Create root
    tree_size, root_idx = allocate_node(tree_size, nodes, x_min, x_max, y_min, y_max)

    # Add all particles
    for i in range(len(points)):
        tree_size = add_particle(tree_size, i, root_idx, points, nodes)

    return root_idx, tree_size
  

# @njit
def step_verlet_vectorized(points):
    """Update all particles at once using NumPy vectorization"""
    tmp_x = points['x'].copy()
    tmp_y = points['y'].copy()
    
    points['x'] = 2 * points['x'] - points['x_prev'] + points['fx'] * TIME_DELTA**2 / points['m']
    points['y'] = 2 * points['y'] - points['y_prev'] + points['fy'] * TIME_DELTA**2 / points['m']
    
    points['x_prev'] = tmp_x
    points['y_prev'] = tmp_y


@njit
def calculate_force(particle_idx, node_idx, points, nodes, theta_threshold):
    """Calculate force on particle from tree node (iterative Barnes-Hut)"""
    if node_idx < 0:
        return  # No node

    # Use a stack to avoid recursion
    stack = [node_idx]
    point = points[particle_idx]

    while stack:
        current_node_idx = stack.pop()

        if current_node_idx < 0:
            continue

        node = nodes[current_node_idx]

        # Skip if this node contains the particle itself
        if node['particle_idx'] == particle_idx:
            continue

        # If this is a leaf node with a different particle, apply force
        if node['particle_idx'] >= 0:
            apply_force_particle_to_particle(particle_idx, node['particle_idx'], points)
            continue

        # Internal node - check s/d criterion
        dx = node['x'] - point['x']
        dy = node['y'] - point['y']
        d = (dx**2 + dy**2)**0.5
        s = node['x_max'] - node['x_min']

        if d > 0 and s / d < theta_threshold:
            # use center of mass approximation
            apply_force_from_com(particle_idx, current_node_idx, points, nodes)
        else:
            # add children to stack
            for child_idx in node['children']:
                if child_idx >= 0:
                    stack.append(child_idx)


@njit
def apply_force_particle_to_particle(p1_idx, p2_idx, points):
    """Apply force between two particles"""
    if p1_idx == p2_idx:
        return

    p1 = points[p1_idx]
    p2 = points[p2_idx]

    dx = p2['x'] - p1['x']
    dy = p2['y'] - p1['y']
    r_squared = max(dx**2 + dy**2, 1e-13)

    F = GRAVITATIONAL_CONSTANT * p1['m'] * p2['m'] / r_squared
    theta = np.arctan2(dy, dx)

    points[p1_idx]['fx'] += F * np.cos(theta)
    points[p1_idx]['fy'] += F * np.sin(theta)


@njit
def apply_force_from_com(particle_idx, node_idx, points, nodes):
    """Apply force from node's center of mass to particle"""
    node = nodes[node_idx]
    point = points[particle_idx]

    dx = node['x'] - point['x']
    dy = node['y'] - point['y']
    r_squared = max(dx**2 + dy**2, 1e-13)

    F = GRAVITATIONAL_CONSTANT * point['m'] * node['m'] / r_squared
    theta = np.arctan2(dy, dx)

    points[particle_idx]['fx'] += F * np.cos(theta)
    points[particle_idx]['fy'] += F * np.sin(theta)


# @njit
def render_points(img: np.ndarray, points, x_min: float, x_max: float, y_min: float, y_max: float):
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
    for particle in points:
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
            # cv2.putText(img, f"v: {round(particle['x']-particle['x_prev'], 4)}, {round(particle['y']-particle['y_prev'], 4)}",
            #     (TEXT_LEFT, TEXT_TOP+2*textHeight), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 255, 255), 1)
            cv2.putText(img, f"f: {round(particle['fx'], 4)}, {round(particle['fy'], 4)}",
                (TEXT_LEFT, TEXT_TOP+2*textHeight), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 255, 255), 1)

    return img


# @njit
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


def simulate(frames: int):
    """Main simulation loop using pointer-based Barnes-Hut with NumPy"""
    x1, y1 = -1.0, 0.0
    vx1, vy1 = 0.3471168881, 0.5327249454
    x1_prev = x1 - vx1 * TIME_DELTA
    y1_prev = y1 - vy1 * TIME_DELTA

    # Particle 2: position (1, 0), velocity (0.3471168881, 0.5327249454)
    x2, y2 = 1.0, 0.0
    vx2, vy2 = 0.3471168881, 0.5327249454
    x2_prev = x2 - vx2 * TIME_DELTA
    y2_prev = y2 - vy2 * TIME_DELTA

    # Particle 3: position (0, 0), velocity (-0.6942337762, -1.0654498908)
    x3, y3 = 0.0, 0.0
    vx3, vy3 = -0.6942337762, -1.0654498908
    x3_prev = x3 - vx3 * TIME_DELTA
    y3_prev = y3 - vy3 * TIME_DELTA

    POINT_TYPE = np.dtype([
        ('m', np.float64),
        ('x', np.float64),
        ('y', np.float64),
        ('x_prev', np.float64),   # For Verlet
        ('y_prev', np.float64),   # For Verlet
        ('vx', np.float64),       # For Leapfrog
        ('vy', np.float64),       # For Leapfrog
        ('fx', np.float64),
        ('fy', np.float64)
    ], align=True)

    POINTS = np.array([
        (1.0, x1, y1, x1_prev, y1_prev, 0.3471168881, 0.5327249454, 0, 0),
        (1.0, x2, y2, x2_prev, y2_prev, 0.3471168881, 0.5327249454, 0, 0),
        (1.0, x3, y3, x3_prev, y3_prev, -0.6942337762, -1.0654498908, 0, 0),
    ], dtype=POINT_TYPE)

    # POINTS = np.array([
    #     (1.0, 0., 0., 0., 0., 0, 0), 
    #     (1.0, 0., 1., 0., 1., 0, 0), 
    #     (1.0, 0., 2., 0., 2., 0, 0), 
    #     (1.0, 0., 3., 0., 3., 0, 0), 
    #     (1.0, 0., 4., 0., 4., 0, 0), 
    #     (1.0, 0., 5., 0., 5., 0, 0), 
    #     (1.0, 0., 6., 0., 6., 0, 0), 
    #     (1.0, 0., 7., 0., 7., 0, 0), 
    #     (1.0, 0., 8., 0., 8., 0, 0), 
    #     (1.0, 0., 9., 0., 9., 0, 0), 
    #     (1.0, 1., 0., 1., 0., 0, 0), 
    #     (1.0, 1., 1., 1., 1., 0, 0), 
    #     (1.0, 1., 2., 1., 2., 0, 0), 
    #     (1.0, 1., 3., 1., 3., 0, 0), 
    #     (1.0, 1., 4., 1., 4., 0, 0), 
    #     (1.0, 1., 5., 1., 5., 0, 0), 
    #     (1.0, 1., 6., 1., 6., 0, 0), 
    #     (1.0, 1., 7., 1., 7., 0, 0), 
    #     (1.0, 1., 8., 1., 8., 0, 0), 
    #     (1.0, 1., 9., 1., 9., 0, 0), 
    #     (1.0, 2., 0., 2., 0., 0, 0), 
    #     (1.0, 2., 1., 2., 1., 0, 0), 
    #     (1.0, 2., 2., 2., 2., 0, 0), 
    #     (1.0, 2., 3., 2., 3., 0, 0), 
    #     (1.0, 2., 4., 2., 4., 0, 0), 
    #     (1.0, 2., 5., 2., 5., 0, 0), 
    #     (1.0, 2., 6., 2., 6., 0, 0), 
    #     (1.0, 2., 7., 2., 7., 0, 0), 
    #     (1.0, 2., 8., 2., 8., 0, 0), 
    #     (1.0, 2., 9., 2., 9., 0, 0), 
    #     (1.0, 3., 0., 3., 0., 0, 0), 
    #     (1.0, 3., 1., 3., 1., 0, 0), 
    #     (1.0, 3., 2., 3., 2., 0, 0), 
    #     (1.0, 3., 3., 3., 3., 0, 0), 
    #     (1.0, 3., 4., 3., 4., 0, 0), 
    #     (1.0, 3., 5., 3., 5., 0, 0), 
    #     (1.0, 3., 6., 3., 6., 0, 0), 
    #     (1.0, 3., 7., 3., 7., 0, 0), 
    #     (1.0, 3., 8., 3., 8., 0, 0), 
    #     (1.0, 3., 9., 3., 9., 0, 0), 
    #     (1.0, 4., 0., 4., 0., 0, 0), 
    #     (1.0, 4., 1., 4., 1., 0, 0), 
    #     (1.0, 4., 2., 4., 2., 0, 0), 
    #     (1.0, 4., 3., 4., 3., 0, 0), 
    #     (1.0, 4., 4., 4., 4., 0, 0), 
    #     (1.0, 4., 5., 4., 5., 0, 0), 
    #     (1.0, 4., 6., 4., 6., 0, 0), 
    #     (1.0, 4., 7., 4., 7., 0, 0), 
    #     (1.0, 4., 8., 4., 8., 0, 0), 
    #     (1.0, 4., 9., 4., 9., 0, 0), 
    #     (1.0, 5., 0., 5., 0., 0, 0), 
    #     (1.0, 5., 1., 5., 1., 0, 0), 
    #     (1.0, 5., 2., 5., 2., 0, 0), 
    #     (1.0, 5., 3., 5., 3., 0, 0), 
    #     (1.0, 5., 4., 5., 4., 0, 0), 
    #     (1.0, 5., 5., 5., 5., 0, 0), 
    #     (1.0, 5., 6., 5., 6., 0, 0), 
    #     (1.0, 5., 7., 5., 7., 0, 0), 
    #     (1.0, 5., 8., 5., 8., 0, 0), 
    #     (1.0, 5., 9., 5., 9., 0, 0), 
    #     (1.0, 6., 0., 6., 0., 0, 0), 
    #     (1.0, 6., 1., 6., 1., 0, 0), 
    #     (1.0, 6., 2., 6., 2., 0, 0), 
    #     (1.0, 6., 3., 6., 3., 0, 0), 
    #     (1.0, 6., 4., 6., 4., 0, 0), 
    #     (1.0, 6., 5., 6., 5., 0, 0), 
    #     (1.0, 6., 6., 6., 6., 0, 0), 
    #     (1.0, 6., 7., 6., 7., 0, 0), 
    #     (1.0, 6., 8., 6., 8., 0, 0), 
    #     (1.0, 6., 9., 6., 9., 0, 0), 
    #     (1.0, 7., 0., 7., 0., 0, 0), 
    #     (1.0, 7., 1., 7., 1., 0, 0), 
    #     (1.0, 7., 2., 7., 2., 0, 0), 
    #     (1.0, 7., 3., 7., 3., 0, 0), 
    #     (1.0, 7., 4., 7., 4., 0, 0), 
    #     (1.0, 7., 5., 7., 5., 0, 0), 
    #     (1.0, 7., 6., 7., 6., 0, 0), 
    #     (1.0, 7., 7., 7., 7., 0, 0), 
    #     (1.0, 7., 8., 7., 8., 0, 0), 
    #     (1.0, 7., 9., 7., 9., 0, 0), 
    #     (1.0, 8., 0., 8., 0., 0, 0), 
    #     (1.0, 8., 1., 8., 1., 0, 0), 
    #     (1.0, 8., 2., 8., 2., 0, 0), 
    #     (1.0, 8., 3., 8., 3., 0, 0), 
    #     (1.0, 8., 4., 8., 4., 0, 0), 
    #     (1.0, 8., 5., 8., 5., 0, 0), 
    #     (1.0, 8., 6., 8., 6., 0, 0), 
    #     (1.0, 8., 7., 8., 7., 0, 0), 
    #     (1.0, 8., 8., 8., 8., 0, 0), 
    #     (1.0, 8., 9., 8., 9., 0, 0), 
    #     (1.0, 9., 0., 9., 0., 0, 0), 
    #     (1.0, 9., 1., 9., 1., 0, 0), 
    #     (1.0, 9., 2., 9., 2., 0, 0), 
    #     (1.0, 9., 3., 9., 3., 0, 0), 
    #     (1.0, 9., 4., 9., 4., 0, 0), 
    #     (1.0, 9., 5., 9., 5., 0, 0), 
    #     (1.0, 9., 6., 9., 6., 0, 0), 
    #     (1.0, 9., 7., 9., 7., 0, 0), 
    #     (1.0, 9., 8., 9., 8., 0, 0), 
    #     (1.0, 9., 9., 9., 9., 0, 0), 
    # ], dtype=POINT_TYPE)
    NODES = np.empty(MAX_TREE_SIZE, dtype=NODE_TYPE)

    TREE_SIZE = 0
    frame_count = 0

    print(f"Starting simulation with {len(POINTS)} particles")
    print(f"THETA = {THETA}, G = {GRAVITATIONAL_CONSTANT}, dt = {TIME_DELTA}")
    print(f"Physics substeps per frame: {PHYSICS_SUBSTEPS}")

    for foo in range(frames):
        for bar in range(PHYSICS_SUBSTEPS):
            root_idx, TREE_SIZE = build_tree(POINTS, NODES)

            POINTS['fx'] = 0
            POINTS['fy'] = 0

            for i in range(len(POINTS)): # hard to vectorize
                calculate_force(i, root_idx, POINTS, NODES, THETA)

            step_verlet_vectorized(POINTS)

        img = np.zeros((IMAGE_SIZE + 2*IMAGE_BORDER_SIZE, IMAGE_SIZE + 2*IMAGE_BORDER_SIZE))

        # Get rendering bounds
        x_min = POINTS['x'].min()
        x_max = POINTS['x'].max()
        y_min = POINTS['y'].min()
        y_max = POINTS['y'].max()

        img = render_points(img, POINTS, x_min, x_max, y_min, y_max)
        img = draw_scale_indicators(img, x_min, x_max, y_min, y_max)

        # Frame counter
        cv2.putText(img, f"Frame: {str(frame_count).zfill(5)} ({PHYSICS_SUBSTEPS} steps/frame)",
                   (IMAGE_BORDER_SIZE-20, IMAGE_BORDER_SIZE-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
        cv2.putText(img, f"Tree nodes: {TREE_SIZE}",
                   (IMAGE_BORDER_SIZE-20, IMAGE_BORDER_SIZE-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

        cv2.imshow('Particles', img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        frame_count += 1

    print(f'\nSimulation ended')
    print(f'Total frames: {frame_count}')
    print(f'Total physics steps: {frame_count * PHYSICS_SUBSTEPS}')
    print('\nFinal particle states:')
    for i, p in enumerate(POINTS):
        print("  Particle {i}: pos=({px:.2f}, {py:.2f}), "
            "vel=({vx:.2f}, {vy:.2f}), "
            "m={m:.0f}".format(
                i=i,
                px=p['x'],
                py=p['y'],
                vx=(p['x']-p['x_prev'])/TIME_DELTA,
                vy=(p['y']-p['y_prev'])/TIME_DELTA,
                m=p['m']
        ))


if __name__ == '__main__':
    simulate(1000)