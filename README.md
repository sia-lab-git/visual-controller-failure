# Discovering Closed-Loop Failures of Vision-Based Controllers via Reachability Analysis
## Case Study: LB-WaypointNav
### [Project Page](https://vatsuak.github.io/failure-detection/) | [Paper](https://arxiv.org/pdf/2211.02736.pdf)<br>

[Kaustav Chakraborty](https://vatsuak.github.io/),
[Somil Bansal](https://smlbansal.github.io/) <br>
University of Southern California.

## Requirements 
This code was tested in a computer with the following setup:
- Ubuntu 20.04
- python 3
- MatLab 2020b
- Tensorflow 2.10 with GPU support

## Get started
### LB-Waypoint Nav Codebase setup
We will first setup the [LB-Waypoint Nav Codebase](https://github.com/smlbansal/Visual-Navigation-Release).

Add the patches and checkout the ```release_tf2``` branch as follows:
```
cd Visual_Navigation_Release
git checkout release_tf2
cd .. && bash ./setup.sh && cd Visual_Navigation_Release
```

Once inside the folder follow the [Readme file](https://github.com/smlbansal/Visual-Navigation-Release/tree/release_tf2#readme) to setup the codebase. Ensure that all the tests pass.

## PART 1: Generate Controller over the grid

After setting up the LB-Waypoint Nav Codebase create the controller file. For that run:

```
python create_CNN_controls.py --batch-size 11 --num-points 51 51 51 11 11 --goal 15 25
```

<u>NOTE</u>: ```create_CNN_controls.py``` creates the required minibatches which is fed into the CNN to generate the controller commands. The output is a  ```optCtrl.mat``` file that creates a grid that can be consumed by helperOC to generate the BRATs. 
You can set the values for the batch-size and numpoints as per your preference. However make sure that atleast one of the values in the ```num-points``` is divisible by the ```batch-size```. Running the above will produce ```optCtrl.mat``` as a grid with dimensions [51,51,51,11,11]. 

## PART 2: Create the obstacle map signed distance function (sdf) for BRAT computation

To do this from the `Visual_Navigation_Release` directory run:
```
cd ..
python create_obstacles_sdf.py --building_number 1 --goal 15 25
```
This will generate a ```obstaclemap.mat``` file containing the sdf of building 1 with respect to the position `[15,25]`

### This procedure will generate two .mat files(```obstaclemap.mat``` and ```optCtrl.mat```). We are now ready to move to the Matlab section!

## PART 3: SOLVE FOR THE BRAT 

### Step 1: Level Set Toolbox and helperOC download 
To perform the Reachability analysis we need to download [ToolboxLS](https://www.cs.ubc.ca/~mitchell/ToolboxLS/).

Run the following commands:

```
wget https://www.cs.ubc.ca/~mitchell/ToolboxLS/toolboxLS-1.1.1.zip
unzip toolboxLS-1.1.1.zip
```

To compute the BRAT in MATLAB run:
```
compute_BRAT.m
```

This will generate data.mat files containing the Value function for the computed 5D BRAT.

## PART 3: Visualize the BRTs

With the computed value function we are set to visualize the zero level set. <br>
In MATLAB run:
```
vis_BRT.m 
```
This will create a `BRAT_waypointNav.png` file showing the BRAT slices for differnt starting heading of the robot.


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
[Combining Optimal Control and Learning for Visual Navigation in Novel Environments](https://smlbansal.github.io/LB-WayPtNav/) <br>
[Level set toolbox](https://www.cs.ubc.ca/~mitchell/ToolboxLS/) <br>
[helperOC](https://github.com/HJReachability/helperOC)

## Contact
If you have any questions, please feel free to email the authors.
