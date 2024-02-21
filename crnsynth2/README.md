# CRN-SYNTH
A library for creating synthetic cancer registry data.

What it offers:
- **Integration**: integrating generators and metrics from various libraries
- **Process**: a structured approach from raw to synth data, including pre-processing, model training, data generation, and post-processing.
- **Experiment**: experiment with different generators and settings in parallel and save results to disk for later comparison

Hence, this library aims to integrate the best synthetic data libraries and allow for structured experimentation to find the best method for your particular dataset.

## Support integration of other libraries
In the relatively new field of synthetic data generation new methodologies are constantly developed. In order to use the latest advancements, we aim to be interoperable with current and future libraries and allow direct comparison between them. 

We integrate the following libraries

**Generators**
- [Synthcity](https://github.com/vanderschaarlab/synthcity)
- [SDV](https://github.com/sdv-dev/SDV)
- [synthetic_data_generation](https://github.com/daanknoors/synthetic_data_generation)

**Metrics**
- [Synthcity](https://github.com/vanderschaarlab/synthcity) - note requires usage of DataLoaders that might make train/test splits in metrics (which should be consistent with splits made during generation)
- [SDMetrics](https://github.com/sdv-dev/SDMetrics)

Moreover, we add our own custom generators and metrics alongside these libraries.

## Library structure
This library is structured in the following manner:
- experiment/ - run and compare multiple synthesis pipelines in parallel
  - configs/ - parameter configuration for experiment, dataset and generators
  - report/ - run multiple synthesis pipelines or comparisons
- metrics/ - metrics for evaluating the synthetic data with respect to the original data
  - performance/ - comparing the performance of models (e.g. ML, survival analysis, etc)
  - privacy/ - evaluating privacy risk of the synthetic data
  - similarity/ - computing the statistical similarity of the synthetic data 
- generators/ - custom generators that can be used alongside third-party generators
- process/ - define and run a single synthesis pipeline on your dataset
  - pipeline/ - a structured approach from raw to synth
  - data/ - functions to process the original and synthetic data


All code in this library is dataset-agnostic. When applying it to a particular dataset and use case, you inherit all functionality from this library to define your own pipeline and run a set of experiments. 
An example on the adult dataset is provided in the examples/ folder

