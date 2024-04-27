from crnsynth.generators.dummy import DummyGenerator
from crnsynth.generators.marginal import MarginalGenerator
from crnsynth.generators.privbayes import PrivBayesDK
from crnsynth.generators.uniform import UniformGenerator

DEFAULT_GENERATORS = {
    "dummy": DummyGenerator,
    "marginal": MarginalGenerator,
    "privbayes": PrivBayesDK,
    "uniform": UniformGenerator,
}
