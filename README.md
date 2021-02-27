# pyParticlePointTracker
A simple particle filter that utilizes the [farnebäck optical flow](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html) and can be extended by additional features.
See examples for usage.
The farnebäck optical flow restricts the search area in such that you should use fluent videos to track points within. If the jumps are too big for farnebäck to be registrated, then the points will not be moved at all!

## Installation
pip install git+https://github.com/ftl999/pyParticlePointTracker

### Importing
from ParticlePointTracker import ParticlePointTracker, Features -> see examples

## Predefined Features
- RGB Histograms
- HSV Histograms
- Template matching
- Grey level coocurence matrix
- SIFT feature matching

## Adding new Features
Just extend the *ParticleFeature* class to add new features.
Feature instances should store their ground truth and current observed values for each point (I just use dicts for that).
The calculated error should then be normalized between 0..1. All errors will be normalized by default, but unnormalized error values will lead to a very unbalanced behaviour.