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


def left_child_add_prefix(text):
    '''
    left child
    '''
    lines = text.split("\n")
    prefixed = ["+---> " + lines[0]]
    prefixed += ["| " + line for line in lines[1:]]
    return "\n".join(prefixed)


def right_child_add_prefix(text):
    '''
    right child
    '''
    lines = text.split("\n")
    prefixed = ["+---> " + lines[0]]
    prefixed += ["  " + line for line in lines[1:]]
    return "\n".join(prefixed)


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

    def count_nodes_below(self, only_leaves=False):
        '''
        count nodes below
        '''
        left_count = self.left_child.count_nodes_below(only_leaves) \
            if self.left_child else 0

        right_count = self.right_child.count_nodes_below(only_leaves) \
            if self.right_child else 0

        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count
    
    def __str__(self):
        if self.is_root:
            node_repr = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            node_repr = f"node [feature={self.feature}, threshold={self.threshold}]"

        children = []

        if self.left_child:
            children.append(left_child_add_prefix(str(self.left_child)))

        if self.right_child:
            children.append(left_child_add_prefix(str(self.right_child)))

        return "\n".join([node_repr] + children)


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

    def count_nodes_below(self, only_leaves=False):
        """
        Returns 1 because a leaf counts as one node.
        """
        return 1
    
    def __str__(self):
        return (f"-> leaf [value={self.value}]")


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

    def count_nodes(self, only_leaves=False):
        '''
        salam necesiz
        '''
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        return self.root.__str__()