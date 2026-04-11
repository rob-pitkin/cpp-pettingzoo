"""Core classes shared across all MPE environments."""

import numpy as np


class EntityState:
    """Physical state of an entity."""
    def __init__(self):
        self.p_pos = None
        self.p_vel = None


class Entity:
    """Base entity class."""
    def __init__(self, name=""):
        self.name = name
        self.size = 0.050
        self.color = None
        self.state = EntityState()


class Agent(Entity):
    """Agent entity."""
    def __init__(self, name="", silent=True):
        super().__init__(name)
        self.silent = silent


class Landmark(Entity):
    """Landmark entity."""
    def __init__(self, name=""):
        super().__init__(name)


class World:
    """World container for agents and landmarks."""
    def __init__(self):
        self.agents = []
        self.landmarks = []

    @property
    def entities(self):
        return self.agents + self.landmarks


__all__ = ["EntityState", "Entity", "Agent", "Landmark", "World"]
