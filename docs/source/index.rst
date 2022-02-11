:github_url: https://github.com/AstraZeneca/chemicalx

ChemicalX Documentation
=======================

ChemicalX is a deep learning library for drug-drug interaction, polypharmacy
side effect, and synergy prediction. The library consists of data loaders
and integrated benchmark datasets. It also includes state-of-the-art deep
neural network architectures that solve the drug pair scoring task.
Implemented methods cover traditional SMILES string based techniques
and neural message passing based models.

.. code-block:: latex

     >@article{chemicalx,
               arxivId = {2202.05240},
               author = {Rozemberczki, Benedek and Hoyt, Charles Tapley and Gogleva, Anna and Grabowski, Piotr and Karis, Klas and Lamov, Andrej and Nikolov, Andriy and Nilsson, Sebastian and Ughetto, Michael and Wang, Yu and Derr, Tyler and Gyori, Benjamin M},
               month = {feb},
               title = {{ChemicalX: A Deep Learning Library for Drug Pair Scoring}},
               url = {http://arxiv.org/abs/2202.05240},
               year = {2022}
     }

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Package Reference

   modules/root
   modules/pipeline


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Notes


   notes/installation
   notes/introduction
   notes/tutorial
   notes/data_processing
   notes/resources


