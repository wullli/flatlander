from flatland.core.env_observation_builder import ObservationBuilder


class DummyObs(ObservationBuilder):

    def get(self, handle: int = 0):
        return []
