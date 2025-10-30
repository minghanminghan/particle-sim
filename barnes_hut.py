from typing import Self
from pprint import pprint
import json

THETA = 0.5

class LeafNode():
    def __init__(self, x:float, y:float, m:float):
        self.x = x
        self.y = y
        self.m = m
        print(f"created LeafNode: {self}")

    def __repr__(self):
        return f"Leaf: {{ x: {self.x}, y: {self.y}, m: {self.m} }}"


class TreeNode(LeafNode):
    def __init__(self, x_min:float, x_max:float, y_min:float, y_max:float, children: list[LeafNode]=[]):
        # aggregate fields
        self.m = -1
        self.x = -1
        self.y = -1

        # bounds
        self.x_min = x_min
        self.x_mid = (x_max - x_min) / 2
        self.x_max = x_max
        self.y_min = y_min
        self.y_mid = (y_max - y_min) / 2
        self.y_max = y_max

        self.children: list[LeafNode | None] = [None, None, None, None]
        
        for child in children:
            self.add(child)

        print(f"created TreeNode: {self}")

    def add(self, point: LeafNode):
        if point.x <= self.x_mid and point.y <= self.y_mid: # nw
            index = 0
        elif self.x > self.x_mid and self.y <= self.y_mid:  # ne
            index = 1
        elif self.x <= self.x_mid and self.y > self.y_mid:  # sw
            index = 2
        else:                                               # se
            index = 3
        child = self.children[index]

        if type(child) == Self:
            print("add target is type TreeNode")
            child.add(point)

        elif type(child) == LeafNode:
            print("add target is type LeafNode")
            # replace LeafNode with TreeNode and insert both
            tmp = child
            if index == 0:   # nw
                x_min, x_max, y_min, y_max = self.x_min, self.x_mid, self.y_min, self.y_mid
            elif index == 1: # ne
                x_min, x_max, y_min, y_max = self.x_mid, self.x_max, self.y_min, self.y_mid
            elif index == 2: # sw
                x_min, x_max, y_min, y_max = self.x_min, self.x_mid, self.y_mid, self.y_max
            else:            # se
                x_min, x_max, y_min, y_max = self.x_mid, self.x_max, self.y_mid, self.y_max

            self.children[index] = TreeNode(x_min, x_max, y_min, y_max, [tmp, point])
            pass

        else: # target child is None
            print("add target is type None")
            self.children[index] = point

        # update mass
        self._update()

    def _update(self):
        print(f"Updating TreeNode")
        children = [child for child in self.children if child is not None]
        if children:
            self.m = sum([child.m for child in children])
            self.x = sum([child.x * child.m for child in children]) / self.m
            self.y = sum([child.y * child.m for child in children]) / self.m

    def __repr__(self):
        return f"Node: {{ x: {self.x}, y: {self.y}, m: {self.m}, x_min: {self.x_min}, x_max: {self.x_max}, y_min: {self.y_min}, y_max: {self.y_max}, # children: {len([i for i in self.children if i is not None])} }}"


def print_tree(node: TreeNode|LeafNode|None, level=0):
    if node is None:
        return
    indent = "  " * level
    print(f"{indent}- {node}")

    if type(node) == TreeNode:
        for n in node.children:
            print_tree(n, level + 1)


def main():
    points = [
        LeafNode(0, 0, 1),
        LeafNode(1, 0, 1),
        LeafNode(0, 1, 1),
        LeafNode(1, 1, 1),
    ]
    print("Points:")
    pprint(points)

    x_min, x_max, y_min, y_max = 0, 1, 0, 1
    
    root = TreeNode(x_min, x_max, y_min, y_max)
    for i in points:
        root.add(i)
        # pprint(root)

    print("Root")
    print_tree(root)


if __name__ == '__main__':
    main()


'''
Total mass and center of mass calculation:
- m_total = m1 + m2
- x_total = (x1*m1 + x2*m2) / m
- y_total = (y1*m1 + y2*m2) / m
'''

'''
Calculating net force on a particular body:
- Traverse the nodes of the tree, starting from the root
- If the center-of-mass of an internal node is sufficiently far from the body:
    - Approximate the bodies contained in that part of the tree as a single body
        - Position: the group’s center of mass
        - Mass: the group’s total mass
- Else (internal node is not sufficiently far from the body):
    - Recursively traverse each of its subtrees
- To determine if a node is sufficiently far away:
    - Compute s / d
        - s: the width of the region represented by the internal node
        - d: the distance between the body and the node’s center-of-mass
    - Ccompare against θ. If s / d < θ, then the internal node is sufficiently far away.
'''

'''
Barnse-Hut tree construction:
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