# AnimalGAN Project Demo

## Introduction

Welcome to the AnimalGAN project demo. In this demonstration, we will guide you through the process of setting up the
environment, running the code available in this repository on real data, and showcasing the expected output.

## Prerequisites

Before we begin, please ensure you have met the following prerequisites:

- **Conda Installation:** Make sure you have Conda installed on your system. If not, you can install it by following the
  instructions provided [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

## Environment Setup

1. **Create a Conda Environment:** Create a Conda environment specifically for the AnimalGAN project using the provide
   YAML file `environment.yml`:

   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the Environment:** Activate the Conda environment you just created:

   ```bash
   conda activate AnimalGAN
   ```

The time required for installation and environment setup may vary based on factors such as your internet speed and
computer performance. Typically, Conda installation takes only a few minutes, encompassing the download, installation,
and configuration of the base Conda environment.

In my experience, recreating the environment using the provided YAML file took approximately 15 minutes. However, these
times are approximate and can differ depending on individual circumstances. If you have a faster internet connection and
a powerful computer, the setup process may be quicker.

## Running the Code

Now that we have our Conda environment set up, let's run the AnimalGAN code on real data. Follow these steps:

1. **Data Preparation:**
   Start by preparing your dataset. Place the dataset files in the designated folder `Data`. We
   have provided the small size example data there to show you the format and help you get started.<br><br>

   When developing AnimalGAN, numeric molecular descriptors are required. For this study, we employed a set of 2D and 3D
   molecular descriptors generated by Mordred. When using AnimalGAN generate clinical pathology measurements for a given
   treatment condition (the combination of compound structure, sacrifice period and dose level), the Mordred descriptors
   of the given compound are also needed.<br><br>

   You can calculate molecular descriptors from the Structure-Data File (SDF) of the compound of interest using the
   provided Python script `calMolDes.py` as a reference.

2. **Training:**
    - Start training the AnimalGAN model on your dataset using the following command:

      ```bash
      python train.py
      ```

    - You can monitor the training progress and metrics during the process.

    - Additionally, you can plot the losses and generated valid ratios using the following command:

      ```bash
      python plotLoss.py
      ```