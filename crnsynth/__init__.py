from synthcity.plugins import Plugins

from crnsynth.synth.custom_generators.marginal_dk import MarginalDK
from crnsynth.synth.custom_generators.privbayes_dk import PrivBayesDK
from crnsynth.synth.custom_generators.uniform_dk import UniformDK

# assign custom generators to synthcity
Plugins().add("privbayes-dk", PrivBayesDK)
Plugins().add("marginal-dk", MarginalDK)
Plugins().add("uniform-dk", UniformDK)
