# STN-Neuron

This is the NEURON python model for the optimized STN multicompartment neuron. The MatingPool.pickle is the pool optimized by a customed genetic algorithm.

## Using the code

`stn_neuron.py` is for model using GW morphology and has is supposed to work with `sim_neuron.py` for parameter value and distribution assignment.

`stn_detail.py` is for detailed morphologies from Chu et al., 2017. They are supposed to work with `sim_detail.py`. Morphology filename is assigned in `sim_detail.py`