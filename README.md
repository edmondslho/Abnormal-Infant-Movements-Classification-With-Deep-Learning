# NETIVAR: NETwork Information Visualization based on Augmented Reality

An Augmented Reality (AR) application which assists users in performing network tasks

![GitHub Logo](/images/AR_example.jpg)
![GitHub Logo](/images/AR_example_2.jpg)

Video Demo on YouTube

[![Vidoe Demo](https://img.youtube.com/vi/tn7IK1dfA_g/0.jpg)](https://www.youtube.com/watch?v=tn7IK1dfA_g)


Two versions of the application are included, One that works deployed to a smartphone that will connect via Telnet to a Cisco 2960 24 port switch, one that only works in the Unity editor that will connect via both SSH and Telnet to TP Link TL-SG2008 8 port switch

Other switches will work, but they need configuring in the Unity editor. Work on the SSH version was dropped when it was found that the feature does not work with android, the android version is more up to date and feature complete.

The main code for the applications is contained in /Assets/Appmanager.cs

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

    @inproceedings{flinton2018NETIVAR,
      author = {Christopher Flinton and Philip Anderson and Hubert P. H. Shum and Edmond S. L. Ho},
      booktitle = {International Conference on Software, Knowledge, Information Management and Applications (SKIMA)},
      title = {NETIVAR: NETwork Information Visualization based on Augmented Reality},
      year = {2018}
    }
    

# Authors and Contributors
NETIVAR is developed by Christopher Flinton (christophfl@hotmail.co.uk) and the proejct is supervised by Edmond Ho (edmond@edho.net). Currently, it is being maintained by Edmond Ho.

# License
NETIVAR is freely available for free non-commercial use, and may be redistributed under these conditions. Please, see the license for further details. Interested in a commercial license? Contact Edmond Ho (edmond@edho.net).
