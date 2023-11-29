from synthcity.plugins import Plugins

from crnsynth.synth.custom_generators.privbayes_dk import PrivBayesDK

# assign custom generators to synthcity
Plugins().add("privbayes-dk", PrivBayesDK)
