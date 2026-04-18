
.. raw:: html

   <div style="text-align: center; margin-top: 20px; margin-bottom: 20px;">

     <!-- LOGO -->
     <a href="https://carcara.readthedocs.io/">
       <img src="_static/logo_light.png" width="500px" class="only-light">
       <img src="_static/logo_dark.png" width="500px" class="only-dark">
     </a>

     <!-- TAGLINE -->
     <p style="font-size: 1.2em; margin-top: 10px;">
       Computational Materials Science and Machine Learning for the Energy Transition
     </p>

     <!-- BADGES -->
     <p>

       <!-- Read the Docs -->
       <a href="https://carcara.readthedocs.io/">
         <img src="https://readthedocs.org/projects/carcara/badge/?version=latest" alt="docs">
       </a>

       <!-- PyPI -->
       <a href="https://pypi.org/project/carcara/">
         <img src="https://img.shields.io/pypi/v/carcara.svg" alt="pypi">
       </a>

       <!-- License -->
       <a href="https://github.com/seixas-research/carcara/blob/main/LICENSE">
         <img src="https://img.shields.io/github/license/seixas-research/carcara" alt="license">
       </a>

       <!-- CI -->
       <a href="https://github.com/seixas-research/carcara/actions">
         <img src="https://img.shields.io/github/actions/workflow/status/seixas-research/carcara/ci.yml" alt="ci">
       </a>

     </p>

   </div>

**Carcará** is a high-performance Python framework designed for atomistic simulations powered by on-the-fly machine learning interatomic potentials (OTF-MLIP). It streamlines the integration of first-principles accuracy with the efficiency of classical force fields, enabling the automated development of robust potentials during the simulation process.

|

**Key Features**

* **On-the-Fly Training**: Automate the training cycle during simulations, reducing the need for manual dataset curation and ensuring the potential is accurate for the relevant phase space.


* **Diverse Configuration Sampling**: Generate new training structures through multiple pathways:

  - **Random Displacements**: Introduce displacements in atomic positions and lattices randomly using "Normal" or "Uniform" distributions.

  - **Molecular Dynamics (MD)**: Structural exploration via various ensembles, such as NVT or NPT.

  - **Minimum Energy Paths (MEP)**: Sampling through diffusion paths and transition states, using methods like Nudged Elastic Band (NEB), getting configurations along reaction coordinates for rare event sampling. 


* **Active Learning & Uncertainty Quantification**: Utilize Machine Learning Committees (ensembles) to identify configurations with high model disagreement, targeting high-uncertainty regions for further training.


* **Scalable Sample Generation**: Efficiently produce large-scale datasets for ML training through continuous random displacements or extended MD trajectories.

|

**Core Focus**

The hallmark of Carcará is its optimization for continuous, autonomous model evolution. Whether you are running long-scale molecular dynamics or generating vast structural libraries via random displacements, Carcará ensures that your machine learning model adapts to the chemical environment in real-time, significantly lowering the barrier for complex materials modeling.

.. toctree::
   :maxdepth: 2
   :hidden:

   Home <self>
   GitHub <https://github.com/seixas-research/carcara>
   PyPI <https://pypi.org/project/carcara/>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   installation
   tutorial/index
   cli
   about