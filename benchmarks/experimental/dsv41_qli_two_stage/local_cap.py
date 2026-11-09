"""Use the existing per-tile candidate scatter to construct score caps."""

from dataclasses import dataclass


@dataclass
class LocalCap:
    work: object
    candidates: object = None

    @classmethod
    def allocate(cls, work):
        return cls(work)

    @property
    def workspace_bytes(self):
        return 0

    def prepare(self, candidates):
        self.candidates = candidates

    def select(self, layer):
        return self.work.select(layer, 10, self.candidates)
