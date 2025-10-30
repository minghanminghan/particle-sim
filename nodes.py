TIME_DELTA = 0.0001

class LeafNode():
    def __init__(self, x:float, y:float, dx: float = 0, dy: float = 0, m:float = 1):
        self.x = x
        self.y = y
        self.x_prev = x
        self.y_prev = y
        self.dx = dx
        self.dy = dy
        self.m = m
        self.fx = 0
        self.fy = 0
        # print(f"created LeafNode: {self}")

    def _step_euler(self): # _update?
        # self.dx = max(-MAX_V, min(MAX_V, self.fx * TIME_DELTA / self.m))
        # self.dy = max(-MAX_V, min(MAX_V, self.fy * TIME_DELTA / self.m))
        self.dx += self.fx * TIME_DELTA / self.m
        self.dy += self.fy * TIME_DELTA / self.m
        self.x += self.dx * TIME_DELTA
        self.y += self.dy * TIME_DELTA
    
    def _step_verlet(self):
        '''
        P_{n+1} = 2*P_n - P_{n-1} + a_n + TIME_DELTA**2
        '''
        tmp_x, tmp_y = self.x, self.y
        self.x = 2 * self.x - self.x_prev + self.fx * TIME_DELTA ** 2 / self.m
        self.y = 2 * self.y - self.y_prev + self.fy * TIME_DELTA ** 2 / self.m
        self.x_prev, self.y_prev = tmp_x, tmp_y
        self.dx = (self.x - self.x_prev) / TIME_DELTA
        self.dy = (self.y - self.y_prev) / TIME_DELTA

    # maybe _step_leapfrog?

    def _cleanup(self):
        self.fx = 0
        self.fy = 0

    def __repr__(self):
        return f"Leaf: {{ x: {self.x}, y: {self.y}, dx: {self.dx}, dy: {self.dy}, m: {self.m}, fx: {self.fx}, fy: {self.fy} }}"


class TreeNode():
    def __init__(self, x_min:float, x_max:float, y_min:float, y_max:float, children: list[LeafNode]=[]):
        # aggregate fields
        self.m = -1
        self.x = -1
        self.y = -1

        # bounds
        self.x_min = x_min
        self.x_mid = x_min + (x_max - x_min) / 2
        self.x_max = x_max
        self.y_min = y_min
        self.y_mid = y_min + (y_max - y_min) / 2
        self.y_max = y_max

        self.children: list[TreeNode | LeafNode | None] = [None, None, None, None]
        
        for child in children:
            self.add(child)

        # print(f"created TreeNode: {self}")

    def add(self, point: LeafNode):
        '''
        Barnes-Hut tree construction:
        - To construct the Barnes-Hut tree, insert the bodies one after another
        - To insert a body b into the tree rooted at node x, use the following recursive procedure:
            1. If node x does not contain a body, put the new body b here.
            2. If node x is an internal node, update the center-of-mass and total mass of x.
            Recursively insert the body b in the appropriate quadrant.
            3. If node x is an external node, say containing a body named c, then there are two bodies b and c in the same region.
            Subdivide the region further by creating four children. 
            Then, recursively insert both b and c into the appropriate quadrant(s).
            Since b and c may still end up in the same quadrant, there may be several subdivisions during a single insertion.
            Finally, update the center-of-mass and total mass of x.
        '''
        if point.x <= self.x_mid and point.y <= self.y_mid:  # nw
            index = 0
        elif point.x > self.x_mid and point.y <= self.y_mid: # ne
            index = 1
        elif point.x <= self.x_mid and point.y > self.y_mid: # sw
            index = 2
        else:                                                # se
            index = 3
        child = self.children[index]

        if isinstance(child, TreeNode):
            # print("current child is TreeNode")
            child.add(point)

        elif isinstance(child, LeafNode):
            # print("current child is LeafNode")
            # replace LeafNode with TreeNode and insert both
            tmp = child
            if index == 0: # nw
                x_min, x_max, y_min, y_max = self.x_min, self.x_mid, self.y_min, self.y_mid
            elif index == 1: # ne
                x_min, x_max, y_min, y_max = self.x_mid, self.x_max, self.y_min, self.y_mid
            elif index == 2: # sw
                x_min, x_max, y_min, y_max = self.x_min, self.x_mid, self.y_mid, self.y_max
            else: # se
                x_min, x_max, y_min, y_max = self.x_mid, self.x_max, self.y_mid, self.y_max

            self.children[index] = TreeNode(x_min, x_max, y_min, y_max, [tmp, point])

        else: # target child is None
            # print("current child is None")
            self.children[index] = point

        # update mass, center of mass
        self._update()

    def __repr__(self):
        return f"Node: {{ x: {self.x}, y: {self.y}, m: {self.m}, x_min: {self.x_min}, x_mid: {self.x_mid}, x_max: {self.x_max}, y_min: {self.y_min}, y_mid: {self.y_mid}, y_max: {self.y_max}, # children: {len([i for i in self.children if i is not None])} }}"

    def _update(self):
        '''
        Total mass and center of mass calculation:
        - m_total = m1 + m2
        - x_total = (x1*m1 + x2*m2) / m
        - y_total = (y1*m1 + y2*m2) / m
        '''
        # print(f"Updating TreeNode")
        children = [child for child in self.children if child is not None]
        if children:
            self.m = sum([child.m for child in children])
            self.x = sum([child.x * child.m for child in children]) / self.m
            self.y = sum([child.y * child.m for child in children]) / self.m
