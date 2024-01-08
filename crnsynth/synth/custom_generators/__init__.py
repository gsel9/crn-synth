from synthcity.plugins import Plugins

from crnsynth.synth.custom_generators.dummy import DummySampler
from crnsynth.synth.custom_generators.marginal_dk import MarginalDK
from crnsynth.synth.custom_generators.privbayes_dk import PrivBayesDK
from crnsynth.synth.custom_generators.uniform_dk import UniformDK

# assign custom generators to synthcity
Plugins().add("privbayes-dk", PrivBayesDK)
Plugins().add("marginal-dk", MarginalDK)
Plugins().add("uniform_sampler_custom", UniformDK)
Plugins().add("dummy_sampler_custom", DummySampler)
