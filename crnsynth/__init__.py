from synthcity.plugins import Plugins

from crnsynth.generators.privbayes_dk import PrivBayesDK

# assign custom generators to synthcity
Plugins().add("privbayes-dk", PrivBayesDK)
