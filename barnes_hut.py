from pprint import pprint
from enum import Enum
import numpy as np
import cv2
from nodes import LeafNode, TreeNode, TIME_DELTA


THETA = 0.5
GRAVITATIONAL_CONSTANT = 1
PHYSICS_SUBSTEPS = 1000  # Number of physics steps per render frame
FRICTION = 0 # not being used
IMAGE_BOTTOM_LEFT = (0, 0)  # (x, y) coordinates of bottom-left corner
IMAGE_TOP_RIGHT = (400, 400)  # (x, y) coordinates of top-right corner
IMAGE_SIZE = 600
IMAGE_BORDER_SIZE = 150


# class Point(Enum):
#     x = 0
#     y = 1
#     x_prev = 2
#     y_prev = 3
#     fx = 4
#     fy = 5
#     m = 6


# def step_verlet(index: int): # parallelize with this
#     p = points[index]
#     tmp_x, tmp_y = p[Point.x], p[Point.y]
#     p[Point.x] = 2 * p[Point.x] - p[Point.x_prev] + p[Point.fx] * TIME_DELTA ** 2 / p[Point.m]
#     p[Point.y] = 2 * p[Point.y] - p[Point.y_prev] + p[Point.fy] * TIME_DELTA ** 2 / p[Point.m]
#     p[Point.x_prev], p[Point.y_prev] = tmp_x, tmp_y
#     p[Point.fx], p[Point.fy] = 0, 0


# points = np.array([
#         (100, 100, 100 - 0.306893, 100 - 0.125507, 0, 0, 1e3),
#         (300, 100, 100 - 0.306893, 100 - 0.125507, 0, 0, 1e3),
#         (200, 200, 100 + 0.613786, 100 + 0.251014, 0, 0, 1e3),
#     ],
#     dtype=[('x', np.float32), ('y', np.float32), ('x_prev', np.float32), ('y_prev', np.float32), ('fx', np.float32), ('fy', np.float32), ('m', np.uint8)])


def calculate_force(node1: LeafNode, node2: LeafNode | TreeNode):
    '''
    To calculate the net force acting on body b, use the following recursive procedure, starting with the root of the quad-tree:
    1. If the current node is an external node (and it is not body b):
        - Calculate the force exerted by the current node on b
        - Add this amount to b’s net force
    2. Otherwise, calculate the ratio s/d
        - If s/d < θ (treat this internal node as a single body):
            - Calculate the force it exerts on body b
            - Add this amount to b’s net force
        - Else:
            - Run the procedure recursively on each of the current node’s children
    '''
    if isinstance(node2, TreeNode): # calculate s/d then decide
        s = node2.x_max - node2.x_min                                       # width of TreeNode region
        d = ((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2) ** 0.5    # distance between self & TreeNode's center of mass
        if d != 0 and s / d < THETA:
            _calculate_force(node1, node2)
        else:
            for i in node2.children:
                if i is not None:
                    calculate_force(node1, i)
    elif node2 != node1: # other is LeafNode
        _calculate_force(node1, node2)

        
def _calculate_force(node1: LeafNode, node2: LeafNode | TreeNode):
    '''
    Use F = G(m_1 * m_2) / r**2 to calculate force exerted on LeafNode
    Then use F = ma update self.dx, self.dy
    '''
    r_squared = max(((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2), 1e-13) # clip for divide by 0
    theta = np.atan2(node2.y - node1.y, node2.x - node1.x)
    F = GRAVITATIONAL_CONSTANT * (node1.m * node2.m) / r_squared
    # node1.fx += min(F * np.cos(theta), MAX_FORCE) # gravitational attraction
    # node1.fy += min(F * np.sin(theta), MAX_FORCE)
    node1.fx += F * np.cos(theta)
    node1.fy += F * np.sin(theta)


def print_tree(node: TreeNode|LeafNode|None, level=0):
    if node is None:
        return
    indent = "  " * level
    print(f"{indent}- {node}")

    if isinstance(node, TreeNode):
        for n in node.children:
            print_tree(n, level + 1)


def transform_points(points: list[LeafNode], x_min:float, x_max:float, y_min:float, y_max:float):
    '''
    returns list[(x, y)] of transformed LeafNodes to render
    '''
    # Use the larger of: particle bounds or fixed image bounds
    x_min_final = min(x_min, IMAGE_BOTTOM_LEFT[0])
    x_max_final = max(x_max, IMAGE_TOP_RIGHT[0])
    y_min_final = min(y_min, IMAGE_BOTTOM_LEFT[1])
    y_max_final = max(y_max, IMAGE_TOP_RIGHT[1])

    x_range = x_max_final - x_min_final
    y_range = y_max_final - y_min_final

    coords = [
        (
            int((i.x - x_min_final) * (IMAGE_SIZE) / x_range),
            int((y_max_final - i.y) * (IMAGE_SIZE) / y_range)  # Flip y-axis: higher y values go up
        )
    for i in points]
    return coords


def render_points(img: np.ndarray, points: list[LeafNode], x_min:float, x_max:float, y_min:float, y_max:float):
    render_points = transform_points(points, x_min, x_max, y_min, y_max)
    # pprint(points)
    # pprint(render_points)
    for i, j in zip(points, render_points):
        if 0 <= j[0] <= IMAGE_SIZE and 0 <= j[1] <= IMAGE_SIZE:
            cv2.circle(img, (j[0] + IMAGE_BORDER_SIZE, j[1] + IMAGE_BORDER_SIZE), 2, (255, 255, 255), -1)
            # cv2.arrowedLine(img, (j[0] + IMAGE_BORDER_SIZE, j[1] + IMAGE_BORDER_SIZE), (int(j[0]+i.dx) + IMAGE_BORDER_SIZE, int(j[1]+i.dy) + IMAGE_BORDER_SIZE), (255, 255, 255), 1)
            force_magnitude = np.sqrt(i.fx**2 + i.fy**2)
            if force_magnitude > 0:
                scale = 5 * np.log10(force_magnitude + 1) / force_magnitude
                fx_scaled = int(i.fx * scale)
                fy_scaled = -int(i.fy * scale)  # Flip y-axis for force arrow
            else:
                fx_scaled = fy_scaled = 0

            cv2.arrowedLine(img,
                (j[0] + IMAGE_BORDER_SIZE, j[1] + IMAGE_BORDER_SIZE),
                (j[0] + IMAGE_BORDER_SIZE + fx_scaled, j[1] + IMAGE_BORDER_SIZE + fy_scaled),
                (255, 255, 255), 1
            )

            FONT_SIZE = 0.4
            (textWidth, textHeight), baseline = cv2.getTextSize(f"x: {round(i.x, 4)}, y: {round(i.y, 4)}", cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, 1)            
            TEXT_LEFT = j[0] + IMAGE_BORDER_SIZE + 10
            TEXT_TOP = j[1] + IMAGE_BORDER_SIZE - 2*textHeight + 4
            cv2.putText(img, f"p: {round(i.x, 2)}, {round(i.y, 2)}", (TEXT_LEFT, TEXT_TOP), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 255, 255), 1)
            # cv2.putText(img, f"v: {round(i.dx, 2)}, {round(i.dy, 2)}", (TEXT_LEFT, TEXT_TOP+2*textHeight), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 255, 255), 1)
            cv2.putText(img, f"f: {round(i.fx, 2)}, {round(i.fy, 2)}", (TEXT_LEFT, TEXT_TOP+4*textHeight), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 255, 255), 1)
        cv2.rectangle(img, (IMAGE_BORDER_SIZE-20, IMAGE_BORDER_SIZE-20), (IMAGE_BORDER_SIZE+IMAGE_SIZE+20, IMAGE_BORDER_SIZE+IMAGE_SIZE+20), (255, 255, 255), 1)
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
    points = [
        LeafNode(100, 100, 0.306893, 0.125507, 1e3),
        LeafNode(300, 100, 0.306893, 0.125507, 1e3),
        LeafNode(200, 200, -0.613786, -0.251014, 1e3),
    ]

    count = 0
    physics_step = 0
    # while count < 10000:
    while True:
        # Run multiple physics steps per frame for accuracy
        for _ in range(PHYSICS_SUBSTEPS):
            x_min = min([i.x for i in points])
            x_max = max([i.x for i in points])
            y_min = min([i.y for i in points])
            y_max = max([i.y for i in points])

            root = TreeNode(x_min, x_max, y_min, y_max)
            for i in points:
                i._cleanup()
                root.add(i)
            for i in points:
                calculate_force(i, root)
            for i in points:
                i._step_verlet()
                # i._step_euler()

            physics_step += 1

        # Render once after all physics substeps
        img = np.zeros((IMAGE_SIZE + 2*IMAGE_BORDER_SIZE, IMAGE_SIZE + 2*IMAGE_BORDER_SIZE))

        x_min = min([i.x for i in points])
        x_max = max([i.x for i in points])
        y_min = min([i.y for i in points])
        y_max = max([i.y for i in points])

        img = render_points(img, points, x_min, x_max, y_min, y_max)
        img = draw_scale_indicators(img, x_min, x_max, y_min, y_max)
        cv2.putText(img, f"Frame: {str(count).zfill(5)} ({PHYSICS_SUBSTEPS} steps per frame)", (IMAGE_BORDER_SIZE-20, IMAGE_BORDER_SIZE-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
        cv2.imshow('Particles', img)

        key = cv2.waitKey(1)
        # if chr(key).lower() == 'q':
        #     break
        count += 1
        # print(count)
    
    print(f'Total steps: {count}')
    print('Final')
    pprint(points)


if __name__ == '__main__':
    main()