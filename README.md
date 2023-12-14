# Trash Classification - Edge Deployment, Traning, & Testing
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Eilliw/trash-classification-public">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Trash Classification - Edge Deployment, Traning, & Testing</h3>

  <p align="center">
    For GEEN-1400 or First-Year Engineering Projects at the University of Colorado Boulder, we, Team-51, were tasked with creating a product that had a positive impact on sustainability especially on our campus.
    We came up with the idea of a smart trash can that would sort your trash to help with the problem of waste being disposed of improperly using computer vision. 
    <br />
    <br />
    <b>Below you will find all of the source code used to train and inference our AI model. The Usage section contains many examples and helpful visuals</b>
    <br />
    <a href="https://github.com/Eilliw/trash-classification-public"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Eilliw/trash-classification-public">View Demo</a>
    ·
    <a href="https://github.com/Eilliw/trash-classification-public/issues">Report Bug</a>
    ·
    <a href="https://github.com/Eilliw/trash-classification-public/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#edge-prerequisites">Edge Prerequisites</a></li>
        <li><a href="#edge-installation">Edge Installation</a></li>
        <li><a href="#train-prerequisites">Train Prerequisites</a></li>
        <li><a href="#train-installation">Train Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][rough-prototype-image]](https://example.com)

Here is a depiction of a very rough prototype of our project.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Raspberry-pi][Rasp]][Rasp-url]
* [![Python3.10][Python]][Python-url]
* [![YOLOv8][Ultralytics]][Ultralytics-url]
* [![Nvidia-Triton][Nvidia]][Triton-url]
* <a href="https://universe.roboflow.com/trashclassification-tayqe/trash-vs-recycling-pi-cam">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This repository contains the triton_server submodule that you can use if you wish to inference a remote model. 
In this project, a remote triton server is being used.


If you would like to initilise the git lfs tc-triton-submodule after cloning the repo follow these steps.
* Make sure git lfs is installed
  ```sh
  git lfs install
  ```
* cd into submodule directory
  ```sh
  cd triton_server
  ```
* Initilize local configuration file & clone
  ```sh
  git submodule init
  ```
  ```sh
  git submodule update
  ```
Or you could just make it easy and just run
```sh
git clone  --recurse-submodules https://github.com/Eilliw/trash-classification-public.git
```
### Edge Prerequisites

This is intended to be run on a raspberry pi running on Debian bookworm
* Pi updates
   ```sh
   sudo apt update
   ```
    ```sh
    sudo apt upgrade
    ```
* Pi required packages
  ```sh
  sudo apt install -y python3-picamera2 python3-rpi.gpio
  ```
* libcamera
  Make sure libcamera is up to date and shows a preview from one of the follwing commands
   ```sh
   libcamera-hello
   ```
   ```sh
   libcamera-hello --qt-preview
   ```
* pigpiod
   This package is heavily reccomended since it controls servos with minial twitch
   ```sh
   sudo apt install pigpiod
   ```
### Edge Installation

Follow these instructions to get a local copy up and running. 
Much of the code will have to be changed to fix path issues.

1. Clone the repo
   ```sh
   git clone https://github.com/Eilliw/trash-classification-public.git
   ```
2. cd into repo directory
3. Create python venv
   ```python
   python3 -m venv venv --system-site-packages
   ```
4. Install python dependencies
   ```sh
   source venv/bin/activate
   ```
   ```sh
   pip3 install -r edge_requirements.txt
   ```
5. Edit `bin/run_edge_on_startup.sh` paths
6. Set Roboflow API key in `.env`
   ```env
   ROBOFLOW_API_KEY="YOUR API KEY HERE"
   ```
7. Run either testing script or running script
   ```sh
   bash bin/run_testing.sh
   ```
   ```sh
   bash bin/run_on_startup.sh
   ```
### Train Prerequisites

### Train Installation
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

### Testing GUI examples

  #### Trash
  <img src="https://github.com/Eilliw/trash-classification-public/blob/main/src/images/usage-examples/test-gui-trash-1.png" width="250x"><img src="https://github.com/Eilliw/trash-classification-public/blob/main/src/images/usage-examples/test-gui-trash-2.png" width="250x">

  The dorito bag and the reciept are being classified as trash at 97.6% and 94.6% confidence respectively


  #### Recycle
  <img src="https://github.com/Eilliw/trash-classification-public/blob/main/src/images/usage-examples/test-gui-recycle-1.png" width="250x">
  
  The Celcius can above is being classified as recycling at 97% confidence


  ### Triton Inference Server
  <img src="https://github.com/Eilliw/trash-classification-public/blob/main/src/images/usage-examples/triton-inference-server-container.PNG" width="450x">

  Of the above models trash-classification is being used as our main model as it is trained apon our [dataset](https://universe.roboflow.com/trashclassification-tayqe/trash-vs-recycling-pi-cam) and the current version of our model is version [3](https://github.com/Eilliw/tc-triton-server/blob/main/model_repo/trash-classification/3/best.pt). All models stored in the [tc-triton-server](https://github.com/Eilliw/tc-triton-server/tree/main) are stored in the torchscript format using git lfs.


_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Simple training Notebook
- [ ] Edge inference
    - [x] via Triton Server
    - [ ] Via local tflite model
- [ ] Testing
    - [x] pi camera data collection
    - [x] Auto dataset upload
    - [x] Gui
    - [ ] Voxel Fiftyone integration
- [x] Triton server container script
- [ ] Post Expo
    - [ ] Docker traning container
    - [ ] Auto traning

See the [open issues](https://github.com/Eilliw/trash-classification-public/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under an MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

<!-- Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - willie.chuter@colorado.edu -->
Willie Chuter - willie.chuter@colorado.edu

Project Link: [https://github.com/Eilliw/trash-classification-public](https://github.com/Eilliw/trash-classification-public)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [Roboflow](https://roboflow.com/)
* [Triton Inference Server](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
* [Zerotier](https://www.zerotier.com/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[repo]: trash-classification-public
[username]: Eilliw

[contributors-shield]: https://img.shields.io/github/contributors/Eilliw/trash-classification-public.svg?style=for-the-badge
[contributors-url]: https://github.com/Eilliw/trash-classification-public/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Eilliw/trash-classification-public.svg?style=for-the-badge
[forks-url]: https://github.com/Eilliw/trash-classification-public/network/members
[stars-shield]: https://img.shields.io/github/stars/Eilliw/trash-classification-public.svg?style=for-the-badge
[stars-url]: https://github.com/Eilliw/trash-classification-public/stargazers
[issues-shield]: https://img.shields.io/github/issues/Eilliw/trash-classification-public.svg?style=for-the-badge
[issues-url]: https://github.com/Eilliw/trash-classification-public/issues
[license-shield]: https://img.shields.io/github/license/Eilliw/trash-classification-public.svg?style=for-the-badge
[license-url]: https://github.com/Eilliw/trash-classification-public/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/willie-chuter
[product-screenshot]: images/screenshot.png

[rough-prototype-image]: src/images/transparent-rough-prototype.png

[trash-1]: src/images/usage-examples/test-gui-trash-1.png
[trash-2]: src/images/usage-examples/test-gui-trash-2.png

[recycle-1]: src/images/usage-examples/test-gui-recycle-1.png

[triton-inference-server-container-shell]: src/images/usage-examples/triton-inference-server-container.PNG


[Rasp]: https://img.shields.io/badge/Raspberry--pi(64bit)-e77499?style=for-the-badge&logo=raspberrypi&logoColor=A22846
[Rasp-url]: https://www.raspberrypi.com/software/operating-systems/#raspberry-pi-os-64-bit

[Python]: https://img.shields.io/badge/Python3.10-000000?style=for-the-badge&logo=python&logoColor=#3776AB
[Python-url]: https://www.python.org/downloads/

[Ultralytics]: https://img.shields.io/badge/Ultralytics--YOLOv8-0000FF?style=for-the-badge&logo=pytorch&logoColor=#EE4C2C
[Ultralytics-url]: https://github.com/ultralytics/ultralytics/

[Nvidia]: https://img.shields.io/badge/Nvidia-Triton-808080?style=for-the-badge&logo=nvidia&logoColor=#76B900
[Triton-url]: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver



[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
