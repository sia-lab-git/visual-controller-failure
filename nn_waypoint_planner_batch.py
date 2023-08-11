import tensorflow as tf
from planners.nn_planner_batch import NNPlanner
from trajectory.trajectory import Trajectory, SystemConfig


class NNWaypointPlanner(NNPlanner):
    """ A planner which selects an optimal waypoint using
    a trained neural network. """

    def __init__(self, simulator, params):
        super(NNWaypointPlanner, self).__init__(simulator, params)
        self.waypoint_world_config = SystemConfig(dt=self.params.dt, n=self.params.batch, k=1)
        self.goal_ego_config = SystemConfig(dt=self.params.dt, n=self.params.batch, k=1)
        self.opt_waypt = SystemConfig(dt=params.dt, n=self.params.batch, k=1, variable=True)
        self.opt_traj = Trajectory(dt=params.dt, n=self.params.batch, k=params.planning_horizon, variable=True)

    def optimize(self, start_config):
            """ Optimize the objective over a trajectory
            starting from start_config.
            """
            p = self.params

            model = p.model

            raw_data = self._raw_data(start_config)
            processed_data = model.create_nn_inputs_and_outputs(raw_data)
            
            # Predict the NN output
            nn_output_n13 = model.predict_nn_output_with_postprocessing(processed_data['inputs'],
                                                                        is_training=False)[:, None]
            # Transform to World Coordinates TODO n = batchsize
            waypoint_ego_config = SystemConfig(dt=self.params.dt, n=self.params.batch, k=1,
                                            position_nk2=nn_output_n13[:, :, :2],
                                            heading_nk1=nn_output_n13[:, :, 2:3])
            
            self.params.system_dynamics.to_world_coordinates(start_config,
                                                waypoint_ego_config,
                                                self.waypoint_world_config)

            # to get the LQR based traj
            spline_traj = []
            for iter in range(self.params.batch):
                data = self.control_pipeline.plan(start_config[iter], self.waypoint_world_config[iter])
                # self.opt_traj.assign_from_trajectory_batch_idx(data[2], 0)
                spline_traj.append(Trajectory.copy(data[2]))
            
            optimal_traj = Trajectory.concat_across_batch_dim(spline_traj)

            # Return the optimal trajectory
            data = {'trajectory': optimal_traj.to_numpy_repr(),
                'waypoint': self.waypoint_world_config.to_numpy_repr()}
            return data
