#!/usr/bin/env python3
"""
Decision Tree structure implementation.

This module defines three classes:
    - Node: internal node of the decision tree
    - Leaf: terminal node containing a prediction value
    - Decision_Tree: tree wrapper

Each node has a depth attribute:
    - The root has depth = 0
    - A child of a node at depth k has depth = k + 1

The goal of max_depth_below() is to recursively compute
the maximum depth found in the subtree below a node.
"""

import numpy as np


class Node:
    """
    Represents an internal node of a decision tree.

    Attributes:
        feature (int): Index of the feature used for the split.
        threshold (float): Threshold used to split the data.
        left_child (Node or Leaf): Left subtree.
        right_child (Node or Leaf): Right subtree.
        is_leaf (bool): Indicates whether the node is a leaf.
        is_root (bool): Indicates whether the node is the root.
        sub_population: Subset of data reaching this node.
        depth (int): Depth of the node in the tree.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Recursively computes the maximum depth of all nodes
        in the subtree rooted at this node.

        Returns:
            int: Maximum depth among all descendant nodes
                 (including leaves).
        """
        left_depth = (self.left_child.max_depth_below()
                      if self.left_child else self.depth)

        right_depth = (self.right_child.max_depth_below()
                       if self.right_child else self.depth)

        return max(left_depth, right_depth)


class Leaf(Node):
    """
    Represents a leaf node of the decision tree.

    Attributes:
        value: Predicted class/value at this leaf.
        depth (int): Depth of the leaf in the tree.
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of this leaf.

        Since a leaf has no children, it is the deepest
        node in its own subtree.

        Returns:
            int: Depth of the leaf.
        """
        return self.depth


class Decision_Tree:
    """
    Wrapper class for a decision tree.

    Attributes:
        root (Node): Root node of the tree.
        max_depth (int): Maximum allowed depth.
        min_pop (int): Minimum population per node.
        split_criterion (str): Criterion used for splitting.
        rng: Random number generator.
    """

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random",
                 root=None):
        self.rng = np.random.default_rng(seed)

        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)

        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Computes the maximum depth of the tree.

        Returns:
            int: Maximum depth among all nodes
                 in the decision tree.
        """
        return self.root.max_depth_below()
