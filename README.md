# Discovering Closed-Loop Failures of Vision-Based Controllers via Reachability Analysis
### [Project Page](https://vatsuak.github.io/failure-detection/) | [Paper](https://arxiv.org/pdf/2211.02736.pdf)<br>

[Kaustav Chakraborty](https://vatsuak.github.io/),
[Somil Bansal](https://smlbansal.github.io/) <br>
University of Southern California.

This is the main branch of the code accompanying the paper: "Discovering Closed-Loop Failures of Vision-Based Controllers via Reachability Analysis". <br>

## Get started
### Cloning the repository

```
git clone --recurse-submodules -j8 https://github.com/sia-lab-git/visual-controller-failure.git
```
This repository has 2 independent branches for each of the case studies:
- TaxiNet
- WayptNav

### Running the TaxiNet Case Study
```
git checkout TaxiNet
git submodule init && git submodule update
```
This will take you to the TaxiNet branch.

### Running the Learning Based Waypoint Navigation Case Study
```
git checkout WayptNav
git submodule init && git submodule update
```
This will take you to the WayptNav branch.


### Each branch has their independent Readme for setting up the code. Please follow the respective instructions to proceed further.


## Citation
If you find our work useful in your research, please cite:
```
@article{chakraborty2023discovering,
  title={Discovering Closed-Loop Failures of Vision-Based Controllers via Reachability Analysis},
  author={Chakraborty, Kaustav and Bansal, Somil},
  journal={IEEE Robotics and Automation Letters},
  volume={8},
  number={5},
  pages={2692--2699},
  year={2023},
  publisher={IEEE}
}
```

## Contact
If you have any questions, please feel free to email the authors.
