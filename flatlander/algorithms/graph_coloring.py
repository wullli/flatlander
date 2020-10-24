"""colors = ['Red', 'Blue', 'Green', 'Yellow', 'Black']

states = ['Andhra', 'Karnataka', 'TamilNadu', 'Kerala']

neighbors = {}
neighbors['Andhra'] = ['Karnataka', 'TamilNadu']
neighbors['Karnataka'] = ['Andhra', 'TamilNadu', 'Kerala']
neighbors['TamilNadu'] = ['Andhra', 'Karnataka', 'Kerala']
neighbors['Kerala'] = ['Karnataka', 'TamilNadu']

colors_of_states = {}"""
import random
from typing import Dict, Any, List


class GreedyGraphColoring:

    @classmethod
    def _promising(cls, node,
                   neighbors,
                   node_colors,
                   color):
        for neighbor in neighbors.get(node):
            color_of_neighbor = node_colors.get(neighbor)
            if color_of_neighbor == color:
                return False

        return True

    @classmethod
    def _get_color_for_state(cls, node, colors, node_colors, neighbors):
        for color in colors:
            if cls._promising(node=node,
                              color=color,
                              neighbors=neighbors,
                              node_colors=node_colors):
                return color
        return None

    @classmethod
    def color(cls, nodes, neighbors: Dict[Any, Any], colors: List[Any]):
        node_colors = {}
        for node in nodes:
            color = cls._get_color_for_state(node=node,
                                             colors=colors,
                                             node_colors=node_colors,
                                             neighbors=neighbors)

            node_colors[node] = color if color is not None else colors[0]

        return node_colors


class ShufflingGraphColoring:

    @classmethod
    def _promising(cls, node,
                   neighbors,
                   node_colors,
                   color):
        for neighbor in neighbors.get(node):
            color_of_neighbor = node_colors.get(neighbor)
            if color_of_neighbor == color:
                return False

        return True

    @classmethod
    def _get_color_for_state(cls, node, colors, node_colors, neighbors):
        for color in colors:
            if cls._promising(node=node,
                              color=color,
                              neighbors=neighbors,
                              node_colors=node_colors):
                return color
        return None

    @classmethod
    def color(cls, nodes, neighbors: Dict[Any, Any], colors: List[Any]):
        node_colors = {}
        keys = list(nodes)
        random.shuffle(keys)
        for k in keys:
            color = cls._get_color_for_state(node=k,
                                             colors=colors,
                                             node_colors=node_colors,
                                             neighbors=neighbors)

            node_colors[k] = color if color is not None else colors[0]

        return node_colors
