# Discovering Closed-Loop Failures of Vision-Based Controllers via Reachability Analysis
## Case Study: TaxiNet
### [Project Page](https://vatsuak.github.io/failure-detection/) | [Paper](https://arxiv.org/pdf/2211.02736.pdf)<br>

[Kaustav Chakraborty](https://vatsuak.github.io/),
[Somil Bansal](https://smlbansal.github.io/) <br>
University of Southern California.

## Requirements 
This code was tested in a computer with the following setup:
- Ubuntu 20.04
- python 3
- MatLab 2020b

## Get started
### Conda Setup
Set up a conda environment with all dependencies like so:
```
conda create -n xplane pip
conda activate xplane
pip install -r requirements.txt
```
### Level Set Toolbox download 
To perform the Reachability analysis we need to download [ToolboxLS](https://www.cs.ubc.ca/~mitchell/ToolboxLS/).

### Simulator setup
To run this code, we need the X Plane simulator. To do so, please follow the steps from [here](https://github.com/StanfordASL/NASA_ULI_Xplane_Simulator/tree/main/src#getting-set-up-with-x-plane-11-for-controller-in-the-loop-simulations)

## PART 1: Generate Controller

Generate controller for morning case (please make sure the Xplane sim is running with the the correct camera config):
```
python gen_controller_morning_clear.py
```

Generate controller for night case:
```
python gen_controller_night_clear.py
```

### This will generate two .mat files where the control commands are stored

## PART 2: SOLVE FOR THE BRT 

### Setup MatLab
1. Open Matlab 
2. Add the `BRT_compuation folder` to path
3. Add the downloaded `ToolboxLS` to path

To generate BRT for morning case run the MatLab Script:
```
brt_morning_clear.m
```

To generate BRT for night case run the MatLab Script:
```
brt_night_clear.m
```

To generate BRT for the ideal case run the MatLab Script:
```
brt_true.m
```

This will generate three .mat files containing the Value functions for the respective case

## PART 3: Visualize the BRTs

With the computed value function we are set to visualize the zero level set. <br>
Open the ``vis_BRT.m`` file and modify the following lines to point to the value fuction (created in Part 2) you wish to visualize.

```
% specify the location of the .mat file with the solved value function
value_function = "<path to value funtion>";
```

Run:
```
vis_BRT.m 
```



## Citation
If you find our work useful in your research, please cite:
```
@article{chakraborty2022discovering,
  title={Discovering Closed-Loop Failures of Vision-Based Controllers via Reachability Analysis},
  author={Chakraborty, Kaustav and Bansal, Somil},
  journal={arXiv preprint arXiv:2211.02736},
  year={2022}
}
```

### Credits:
NASA ULI Xplane Simulator: https://github.com/StanfordASL/NASA_ULI_Xplane_Simulator.git <br>
Level set toolbox: https://www.cs.ubc.ca/~mitchell/ToolboxLS/

## Contact
If you have any questions, please feel free to email the authors.
