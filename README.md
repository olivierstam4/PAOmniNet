# Pa Omninet: A Retraining-Free, Generalizable Deep Learning Framework for Robust Photoacoustic Image Reconstruction


This code was used as part of the work presented in 

Olivier J.M. Stam, Kalloor Joseph Francis* and Navchetan Awasthi* "PA OmniNet: A Retraining-Free, Generalizable Deep Learning
Framework for Robust Photoacoustic Image Reconstruction” Available at SSRN: https://ssrn.com/abstract=5158042 or http://dx.doi.org/10.2139/ssrn.5158042

Please cite this if you find these codes helpful in your work.


* Authors contributed equally

[//]: # (**The raw measurement data for the experimental experiments is not provided and can be requested.)
**Please contact if you find any mistakes or if you need any help regarding the codes. The codes for installation of environement, running the experiments, along with visualization and running the app are provided below for the users.

# Visualization

The codes for visualizing the results for the various datasets are given below:

#Jupyter notebook implementation for creating figures for the OADAT dataset: generalevalOADAT.ipynb \
#Jupyter notebook implementation for creating figures for the OADAT dataset: ThesisimgOADAT.ipynb\
#Jupyter notebook implementation for reformatting mice data: Altermousedata.ipynb\
#Jupyter notebook implementation for evaluating the mice sub-datasets: generalevalmice.ipynb\
#Jupyter notebook implementation for creating figures for the mice sub-datasets: Thesisimgmice.ipynb

# Python Application
The code to build and try out the app are given here:

App: \
#Python implementation for the model used in the backend of the app: App_model.py\
#Python implementation for the backend of the app: PA_OmniNet_backend.py\
#Python implementation for the app: PA_OmniNet_app.py

# Helper Functions

Helper functions:\
#Python implementation for helper functions around all files: helper_functions.py\
#Python implementation for testing functions in evaluation files: testing_functions.py

# Training the models

The codes for training the models for the various datasets are given below with the the helper and testing codes above:

Mice:\
#Python implementation for training the U-nets on the Mice data: Mouse_unet.py

OADAT:\
#Python implementation for training PA OmniNet on SWFD Semi, SWFD Multi and MSFD: PA_OmniNet_training.py\
#Python implementation for training PA OmniNet on SCD: PA_OmniNet_training_SCD.py\
#Python implementation for training U-net on OADAT MSFD: Unet_MFSD_train.py\
#Python implementation for training U-net on OADAT SCD: Unet_SCD_train.py\
#Python implementation for training U-net on OADAT SWFD Semi: Unet_semi_train.py\
#Python implementation for training U-net on OADAT SWFD Multi: Unet_SWFDmulti_train.py
#Python implementation for retrieving numerical results for the OADAT datasets on the U-net: metrics_Unet.py\
#Python implementation for retrieving all the images of SWFD Semi, SWFD Multi and MSFD: PA_OmniNet_all_images.py\
#Python implementation for retrieving all the images of SCD: PA_OmniNet_all_images_SCD.py\
#Python implementation for retrieving numerical results for the OADAT datasets on the PA OmniNet for SWFD Semi, SWFD Multi and MSFD: PA_OmniNet_testing.py\
#Python implementation for retrieving numerical results for the OADAT datasets on the PA OmniNet for SCD: PA_OmniNet_testing_SCD.py\
#Python implementation for  retrieving all the images of the OADAT sub-datasets on the U-net: Unet_all_images.py


Backend models: \
#Python implementation for PA OmniNet testing model: testing_model.py\
#Python implementation for U-net backbone: unet1.py


#Environment install: environment.yml




