# NETIVAR: NETwork Information Visualization based on Augmented Reality

SMARTBabies - Sensing Movement using Action Recognition Technology in Babies

# Introduction
The pursuit of early diagnosis of cerebral palsy has been an active research area with some very promising results using tools such as the General Movements Assessment (GMA). In this project, we explore the feasibility of extracting pose-based features from video sequences to automatically classify infant body movement into two categories, normal and abnormal. The classification was based upon the GMA, which was carried out on the video data by an independent expert reviewer. We explore the viability of using these pose-based feature sets for automated classification within a wide range of machine learning frameworks by carrying out extensive experiments.

![GitHub Logo](/images/AR_example.jpg)


# Requirements
In order to run this application, you must have:

* A Cisco 2960 24 port switch running on the same network as the device, with an address matching the one in the object in Unity (Default is 192.168.0.1) Running Telnet
* All ports must be configured as if it were a fully used switch. (Descriptions etc)
* An android smart-phone running 7.0 Nougat, with a camera and decent CPU and Unity 2017.4.0f1
* A printed off version of the user-defined target

# Getting started

*If using the smartphone*

Launch the application and login with the appropriate username and password, tap login. A notice will appear warning of the dangers of telnet, click continue and move the camera away from the target Move the camera back to the target to display the port information Use the arrow buttons in the top right corner to move between port options, the name of the option will be displayed at the bottom of the application

*If using Unity player (must have a webcam, may need to change Vuforia settings in Unity to setup)*

Make sure the computer is connected to the right switch/ switch is running SSH Press the play button in unity Hold the target in the webcam's view Port information will be displayed Use the arrow buttons in the top right corner to move between port options, the name of the option will be displayed at the bottom of the application

# Citation
Please cite these papers in your publications if it helps your research:

    @ARTICLE{McCay:DeepBaby,
      author={K. D. {McCay} and E. S. L. {Ho} and H. P. H. {Shum} and G. {Fehringer} and C. {Marcroft} and N. D. {Embleton}},
      journal={IEEE Access}, 
      title={Abnormal Infant Movements Classification With Deep Learning on Pose-Based Features}, 
      year={2020},
      volume={8},
      number={},
      pages={51582-51592},
      doi={10.1109/ACCESS.2020.2980269}
    }
    
    @INPROCEEDINGS{McCay:PoseBaby,
       author={K. D. {McCay} and E. S. L. {Ho} and C. {Marcroft} and N. D. {Embleton}},
       booktitle={2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
       title={Establishing Pose Based Features Using Histograms for the Detection of Abnormal Infant Movements},
       year={2019},
       volume={},
       number={},
       pages={5469-5472},
       doi={10.1109/EMBC.2019.8857680}
    }
         

# Authors and Contributors
SMARTBabies is developed by Kevin McCay (kevin.d.mccay@northumbria.ac.uk) and the proejct is supervised by Edmond Ho (edmond@edho.net). Currently, it is being maintained by Edmond Ho.

# License
SMARTBabies is freely available for free non-commercial use, and may be redistributed under these conditions. Please, see the license for further details. Interested in a commercial license? Contact Edmond Ho (edmond@edho.net).
