# Lung X-ray COVID-19 Classification

This repository contains the source code for a Convolutional Neural Network (CNN) that predicts whether a person is infected with COVID-19 or is healthy, based on lung X-ray images. The model is designed to classify X-ray images into two categories: COVID-19 positive and normal.

## Model Overview

The model architecture consists of multiple convolutional layers followed by dense layers, optimized for image classification tasks. Below is a visualization of the CNN architecture used in this project:

![Model Visualization](https://www.kaggleusercontent.com/kf/91044854/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..K6dJSDqxPZjmzAei_OsIMg.DPp2fOY8qm-aymS9ardEJcO0Dw6QpU-ZAAkRzgU7zEy0MDoAPoV87zxzjZlw-IKdbtfS4X0_qdXuwJmlhvuC80ERid93s84TC_M8zmhM6GvRf9Ugpqe7_zo5EuUucxrTMewFHJGolt6BE5kIHefrKYHJ2oSF2rbZBwrJOXsc8pyS9uVfvjpDdEfcn04fOpRnhmc-4YIoUAMWh3A8yXNFeDjo4yjJqak7LFDXhm_oVWnvA8sbZCvJQekXoB0v1qIx-aY7Fat9giGkDopEqkAFTNwiKvUHJqo0qlFTXsHtpf6xLzhlMZaHdfDPmrnEAfU991je6sl3H9UtfD3Bq3eSshpdA5K8SNOBdjahcRojv348eapMScmN0u5kXUZsUly4xaQXxnjzIKxjEYaxZRVKOaH5XkJrSzZTdnuzNn1W_SybFCdvIzvA9Im0RChfhagg8UYTHI3q0JI2bJRRaz6cM-EuMy-KZi6ACU_8Dl0oAJqwWZZXIChbUkBZMxCm2fd6IKz0-_wC2B_JFCwGeLEZu6Kvt8lImIHnCcC_By_JOX1KpdaqxRnvsAHf1LaX7gTAFTuxXx2Ef7ai5NSkEEeehY_US_uMNK8OwgKKmUCJn4gRlaGSs-E6GtxSlHpA3HFsYhDE8gZLSIPP-_GdD-eprFKCjOlZUMMMnjeN4eY_wjc.eJGvY4T36P_4C21sn3FVGQ/__results___files/__results___20_0.png)

### Key Layers:
- **Rescaling**: Normalizes pixel values.
- **Conv2D**: Extracts important features from the images.
- **MaxPooling2D**: Reduces the dimensionality while preserving the important features.
- **Flatten**: Converts the 2D matrix to a 1D vector.
- **Dense**: Performs the final classification.

## Dependencies

The project requires the following Python libraries:

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

You can install these dependencies using pip:

```bash
pip install tensorflow keras numpy pandas matplotlib
