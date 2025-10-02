## Training and inference notebooks

This folder contains notebooks to train and test deep-learning models for source localization. It includes:

- **Training notebooks** for three model variants:
  - **Degrees model**: predicts position angles in degrees (theta, phi).
  - **Sin/Cos model**: predicts labels transformed with sine and cosine of the angles.
  - **Cartesian model**: predicts unit-vector Cartesian coordinates (x, y, z) on the sphere.

- **Inference notebooks** to evaluate trained models and measure localization performance on validation/test data.

- **Plot inference notebooks** that use the plot dataset to generate HEALPix probability maps from model outputs.

All notebooks are located in `dl-loc/train_and_test/`.


