import time
import os

import numpy as np
import pybullet as p
import pybullet_data
import tacto

from utilities import Models, Camera
from tqdm import tqdm
from thing import Thing
from scipy.spatial import cKDTree



class ClutteredPushGrasp:
    # Global constants
    SIMULATION_STEP_DELAY = 1 / 25000.


    def __init__(self, robot, models: Models, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis

        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera

        # Define Pybullet simulation environment
        # p.GUI for a simulation w/ GUI, p.DIRECT for a headless simulation
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.planeID = p.loadURDF("plane.urdf")

        # Import the robot
        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # Prepare parameters
        self.resetUserDebugParameters()
        self.robot.load_digit_parm()        # Read the parameters.yaml file to load robot parameters

        # Mount DIGIT tactile sensors and cameras
        self.digits = tacto.Sensor(**self.robot.tacto_info, background = self.robot.bg)
        p.resetDebugVisualizerCamera(**self.robot.camera_info)
        self.digits.add_camera(self.robot.id, self.robot.link_ID)

        # Load the target object defined in the parameters.yaml file (or other objects later into the envrionment)
        self.object_name        = self.robot.object_info["object_name"]
        self.object_position    = self.robot.object_info["object_position"]
        self.global_scaling     = self.robot.object_info["global_scaling"]
        self.container = Thing(self.object_name, self.object_position, self.global_scaling)

        # Load the specified object to the DIGIT tactile sensors
        self.digits.add_object(self.container.urdf_path, self.container.ID, self.container.objectScale)

        # Load robot parameters
        self.Z_PADDING          = self.robot.robot_info["z_padding"]
        self.VELOCITY_SCALE     = self.robot.robot_info["velocity_scale"]

        # Load data collection parameters
        self.RANDOM_POSES_COUNT = self.robot.data_info["poses_count"]

    # Hooks to pybullet.stepSimulation()
    def step_simulation(self):
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    # Auxiliary function for step_simulation with predetermined step size
    def fixed_step_sim(self, step_size):
        for _ in range(step_size):
            self.step_simulation()

    # Read the values of the sliders in the simulation GUI
    def read_debug_parameter(self):   
        # read the value of task parameter        
        x = p.readUserDebugParameter(self.xin)        
        y = p.readUserDebugParameter(self.yin)        
        z = p.readUserDebugParameter(self.zin)       
        roll = p.readUserDebugParameter(self.rollId)        
        pitch = p.readUserDebugParameter(self.pitchId)        
        yaw = p.readUserDebugParameter(self.yawId)        
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)
        return x, y, z, roll, pitch, yaw, gripper_opening_length

    # Initialize simulation GUI button values
    def initButtonVals(self):
        self.jointObsButtonVal = 2.0
        self.baselineDataColButtonVal = 2.0
        self.mlpDataColButtonVal = 2.0
        self.getObjectFeaturesButtonVal = 2.0
        self.resetSimulationButtonVal = 2.0
        self.fetch6dButtonVal = 2.0
    
    def readGetObjectFeaturesButton(self):
        if p.readUserDebugParameter(self.getObjectFeaturesButton) >= self.getObjectFeaturesButtonVal:
            object_geometric_features = self.getObjectGeometry(self.container.ID)
        self.getObjectFeaturesButtonVal = p.readUserDebugParameter(self.getObjectFeaturesButton)+ 1.0

    def readJointObsButton(self, angles):
        if p.readUserDebugParameter(self.jointObsButton) >= self.jointObsButtonVal:
            print(f"Joint angles (length={len(angles)}): {angles}")
        self.jointObsButtonVal = p.readUserDebugParameter(self.jointObsButton) + 1.0
    
    def readResetSimulationButton(self):
        if p.readUserDebugParameter(self.resetSimulationButton) >= self.resetSimulationButtonVal:
            self.reset_simulation()
        self.resetSimulationButtonVal = p.readUserDebugParameter(self.resetSimulationButton) + 1.0
    
    def readBaselineDataCollectionButton(self):
        if p.readUserDebugParameter(self.baselineDataColButton) >= self.baselineDataColButtonVal:
            self.collect_data()
        self.baselineDataColButtonVal = p.readUserDebugParameter(self.baselineDataColButton) + 1.0

    def readMLPDataCollectionButton(self):
        if p.readUserDebugParameter(self.mlpDataColButton) >= self.mlpDataColButtonVal:
            self.collect_data_mlp()
        self.mlpDataColButtonVal = p.readUserDebugParameter(self.mlpDataColButton) + 1.0
    
    def readFetch6dButton(self, action):        
        if p.readUserDebugParameter(self.fetch6dButton) >= self.fetch6dButtonVal:           
            print(f"End effector 6D pose: {action}")        
        self.fetch6dButtonVal = p.readUserDebugParameter(self.fetch6dButton) + 1.0


    # HELPER FUNCTIONS FOR SIMULATION
    def gripperCollisionDetection(self):
        """
        Determines if the gripper is in contact with an object.
        """

        # linkIndex=13, 19 for DIGIT sensor on left and right finger respectively
        contact_points_left_sensor = p.getContactPoints(self.robot.id, self.container.ID, linkIndexA=13, linkIndexB=-1)
        contact_points_right_sensor = p.getContactPoints(self.robot.id, self.container.ID, linkIndexA=19, linkIndexB=-1)
        return len(contact_points_left_sensor+contact_points_right_sensor) > 0
    
    def isTactileReadingsValid(self, depth, color):
        """
        Determines if the given depth and color tactile data provide any meaningful results, i.e. the mean is above a 
        certain threshold as a sanity check.

        We use a threshold of 1e-3 to determine the minimal acceptable mean for the tactile data
        """
        av_d0, av_d1 = np.mean(depth[0]), np.mean(depth[1])         # Mean of depth data for each finger
        av_c0, av_c1 = np.mean(color[0]), np.mean(color[1])         # Mean of color data for each finger
        
        # Additionally, we scale av_d0 and av_d1 to the range (0, 255) to match the range
        # of color data
        av_d0, av_d1 = av_d0 * 255, av_d1 * 255
        return all(i >= 1e-3 for i in [av_d0, av_d1, av_c0, av_c1])
    
    # HELPER FUNCTIONS FOR DATA COLLECTION PIPELINE
    # Generate random poses by applying Gaussian noise to a base pose(s)
    def generateGaussianNoisePose(self, pose, object_name):
        """
        Generates a random end effector pose following a Gaussian distribution.
        
        @returns random_poses: filled array of generated end effector poses
        """
        
        # Apply 6d gaussian noise to base hand pose
        sixd_noise = {
            # Baseline noise
            "block": np.insert(np.random.normal(0, 1, 5), 1, 0),

            # MLP noise
            "block1": np.random.normal(0, 1, 6),
            "block2": np.random.normal(0, 1, 6),
            "block3": np.random.normal(0, 1, 6),
            "cylinder1": np.random.normal(0, 1, 6),
            "cylinder2": np.random.normal(0, 1, 6),
            "cylinder3": np.random.normal(0, 1, 6),
            "bottle1": np.random.normal(0, 1, 6),
            "bottle2": np.random.normal(0, 1, 6),
            "bottle3": np.random.normal(0, 1, 6),
        }
        
        # Rescale generated poses to fit scale of simulation
        noisy_poses = pose + sixd_noise[object_name] * 0.01
        noisy_poses[2] += self.Z_PADDING
        return noisy_poses
    
    def execute_pose(self, grasp_pose):
        """
        Executes a grasp given a specific hand pose

        @params grasp_pose: 6d numpy array of an end effector pose (cartesian coordinates (x, y, z) &
          euler angles (r, p, y))
        """

        # 1. Move arm to pose and prepare gripper
        self.robot.manipulate_ee(grasp_pose, 'end', self.VELOCITY_SCALE)

        # 1b. [Temporary fix - Reset object in case the gripper collides with the object unexpectedly]
        self.fixed_step_sim(500)
        self.container.resetObject()
        self.fixed_step_sim(1000)

        self.robot.open_gripper()
        self.fixed_step_sim(500)

        # 2. Lower the arm by z=2
        lower_sixd_pose = grasp_pose.copy()
        lower_sixd_pose[2] -= self.Z_PADDING
        self.robot.manipulate_ee(lower_sixd_pose, 'end', self.VELOCITY_SCALE)
        self.fixed_step_sim(500)

        # 3. Close gripper to perform grasp
        self.robot.close_gripper()
        self.fixed_step_sim(200)

        # 4. Determine if gripper is in stable contact with object (measure over 500 steps)
        isGripperInContactTimeframe1 = self.gripperCollisionDetection()
        self.fixed_step_sim(500)
        isGripperInContactTimeframe2 = self.gripperCollisionDetection()

        # 5. Gripper contact is stable - record grasp and tactile data
        if isGripperInContactTimeframe1 is True and isGripperInContactTimeframe2 is True:
            # 6. Record depth and color tactile data
            self.digit_step()
            self.fixed_step_sim(200)
            color = np.asarray(self.color)      # (2, 160, 120, 3)
            depth = np.asarray(self.depth)      # (2, 160, 120)

            # 7. Tactile data sanity check
            if self.isTactileReadingsValid(depth, color) is True:
                # 7. Lift object for 5s to determine successful vs unsuccessful grasp
                initial_object_z_pos = self.container.getPos()[2]
                self.robot.manipulate_ee(grasp_pose, 'end', self.VELOCITY_SCALE)
                self.fixed_step_sim(200)

                # 8. Record success/failure by determining if object moved after 1500 steps
                self.fixed_step_sim(750)
                final_object_z_pos = self.container.getPos()[2]

                # 9. Determine grasp success
                grasp_outcome = True if final_object_z_pos - initial_object_z_pos > 0 else False
                return (depth, color, grasp_outcome)
        
        # After grasp execution, release object and move robot to specific pose to prevent collision
        # self.robot.open_gripper()
        # self.robot.manipulate_ee(grasp_pose, 'end', self.VELOCITY_SCALE)
        return None

    def save_dataset(self, dataset_filename, folder_name, dataset):
        CURR_DIR = os.getcwd()
        TARGET_DIR = "./src/" + folder_name

        dataset_path = os.path.join(CURR_DIR, TARGET_DIR, dataset_filename)
        np.save(dataset_path, dataset)
        print(f"Dataset saved to {dataset_path}")

    def collect_data(self):
        """
        This function executes a data collection loop by generating N gaussian-distributed end effector poses,
        then collecting the corresponding DIGIT tactile sensor readings on each finger of the gripper (as the 
        tactile data) as well as the end effector poses (as the visual data).

        The collected data is then used for stability classification in which the project aims to find the
        best representation of this data. This serves as the basis for further work on learning a generative
        model.
        """      

        # Arrays to populate when collecting data
        end_effector_poses = np.empty((0, 6))                # End effector is a 6D structure
        tactile_depth_data = np.empty((0, 2, 160, 120))      # Depth data (160x120) per finger (x2)
        tactile_color_data = np.empty((0, 2, 160, 120, 3))   # Color data (160x120x3) per finger (x2)
        grasp_outcomes     = np.empty(0)

        # Counters for logging
        success_count = 0
        failure_count = 0

        # Base hand poses for different objects will vary as a result of manual trials on these objects.
        base_hand_poses = {
            "block": [(0.0, -0.014147371053695679, 0.19894737422466278, -3.140000104904175, 1.570796251296997, 1.5707963705062866),
                (0.0, -0.0047158002853393555, 0.19894737422466278, -3.140000104904175, 1.570796251296997, 1.5707963705062866),
                (0.0, 0.021221041679382324, 0.19894737422466278, -3.140000104904175, 1.570796251296997, 1.5707963705062866),]
        }

        try:
            object_hand_poses = base_hand_poses[self.object_name]
            for seed_pose in object_hand_poses:
                print(f"Starting seed pose simulation: {seed_pose}")

                while success_count < self.RANDOM_POSES_COUNT or failure_count < self.RANDOM_POSES_COUNT:
                    # Reset robot and arm only
                    self.reset_simulation()
                    self.fixed_step_sim(1000)
                    noisy_pose = self.generateGaussianNoisePose(seed_pose, self.object_name)
                    grasp_data = self.execute_pose(noisy_pose)

                    if grasp_data is not None:
                        depth, color, grasp_is_good = grasp_data

                        # Only record the grasp data if the tactile data is valid
                        if depth is None or color is None:
                            print(f"Not saving grasp data to dataset :(")
                        else:
                            if grasp_is_good is True and success_count < self.RANDOM_POSES_COUNT:
                                # Save recorded data to corresponding datasets
                                end_effector_poses = np.append(end_effector_poses, np.array([noisy_pose]), axis=0)
                                tactile_depth_data = np.append(tactile_depth_data, np.array([depth]), axis=0)
                                tactile_color_data = np.append(tactile_color_data, np.array([color]), axis=0)
                                grasp_outcomes = np.append(grasp_outcomes, np.ones(shape=(1,)), axis=0)
                                success_count += 1
                                print(f"Data analysed and saved - Successes: {success_count} | Failures: {failure_count}")
                            elif grasp_is_good is False and failure_count < self.RANDOM_POSES_COUNT:
                                # Save recorded data to corresponding datasets
                                end_effector_poses = np.append(end_effector_poses, np.array([noisy_pose]), axis=0)
                                tactile_depth_data = np.append(tactile_depth_data, np.array([depth]), axis=0)
                                tactile_color_data = np.append(tactile_color_data, np.array([color]), axis=0)
                                grasp_outcomes = np.append(grasp_outcomes, np.zeros(shape=(1,)), axis=0)
                                failure_count += 1
                                print(f"Data analysed and saved - Successes: {success_count} | Failures: {failure_count}")
                            print(end_effector_poses.shape)

                # Reset counters
                success_count = 0
                failure_count = 0

            folder_name = "baseline_model"
            # Save collected data into .npy files for future loading
            self.save_dataset("depth_ds.npy", folder_name, tactile_depth_data)
            self.save_dataset("color_ds.npy", folder_name, tactile_color_data)
            self.save_dataset("poses_ds.npy", folder_name, end_effector_poses)
            self.save_dataset("grasp_outcomes.npy", folder_name, grasp_outcomes)

        except KeyError:
            print("Baseline model only permits object_name = block. Update the parameters.yaml file and change the 'object_name' parameter to block.")

        
    def collect_data_mlp(self):
        print("MLP data collection started...", flush=True)
        poses = self.load_mlp_poses()

        # Counters for logging
        success_count = 0
        failure_count = 0
        total_poses = 0

        SEED_POSE_COUNT = 20

        end_effector_poses = np.empty((0, 6))                             # End effector is a 6D structure
        tactile_depth_data = np.empty((0, 2, 160, 120))                   # Depth data (160x120) per finger (x2)
        tactile_color_data = np.empty((0, 2, 160, 120, 3))                # Color data (160x120x3) per finger (x2)
        grasp_outcomes     = np.empty(0)

        for seed_pose in poses[self.object_name]:
            print(f"Starting seed pose simulation: {seed_pose}")

            # We need 10 good and 10 bad grasps per seed pose
            while success_count < SEED_POSE_COUNT or failure_count < SEED_POSE_COUNT:
                # 11. Reset robot and arm only
                self.reset_simulation()
                self.fixed_step_sim(1000)
                noisy_pose = self.generateGaussianNoisePose(seed_pose, self.object_name)
                grasp_data = self.execute_pose(noisy_pose)

                if grasp_data is not None:
                    depth, color, grasp_is_good = grasp_data

                    # Record data if tactile data is valid
                    if depth is None or color is None:
                        print(f"Not saving grasp data to dataset :(")
                    else:
                        if grasp_is_good is True and success_count < SEED_POSE_COUNT:
                            # Save recorded data to corresponding datasets
                            end_effector_poses = np.append(end_effector_poses, np.array([noisy_pose]), axis=0)
                            tactile_depth_data = np.append(tactile_depth_data, np.array([depth]), axis=0)
                            tactile_color_data = np.append(tactile_color_data, np.array([color]), axis=0)
                            grasp_outcomes = np.append(grasp_outcomes, np.ones(shape=(1,)), axis=0)
                            success_count += 1
                            total_poses += 1
                            print(f"Data saved - Successes: {success_count} | Failures: {failure_count} | Total: {total_poses}")
                        elif grasp_is_good is False and failure_count < SEED_POSE_COUNT:
                            # Save recorded data to corresponding datasets
                            end_effector_poses = np.append(end_effector_poses, np.array([noisy_pose]), axis=0)
                            tactile_depth_data = np.append(tactile_depth_data, np.array([depth]), axis=0)
                            tactile_color_data = np.append(tactile_color_data, np.array([color]), axis=0)
                            grasp_outcomes = np.append(grasp_outcomes, np.zeros(shape=(1,)), axis=0)
                            failure_count += 1
                            total_poses += 1
                            print(f"Data saved - Successes: {success_count} | Failures: {failure_count} | Total: {total_poses}")
                        print(end_effector_poses.shape)

            print("Collected enough data for seed pose. Moving to next pose")

            # Reset counters
            success_count = 0
            failure_count = 0

        # Save collected data into .npy files for future loading
        # folder_name = "datasets"
        folder_name = "new_datasets"
        file_path = f"{self.object_name}_ds/"
        self.save_dataset(file_path + "depth_ds.npy", folder_name, tactile_depth_data)
        self.save_dataset(file_path + "color_ds.npy", folder_name, tactile_color_data)
        self.save_dataset(file_path + "poses_ds.npy", folder_name, end_effector_poses)
        self.save_dataset(file_path + "grasp_outcomes.npy", folder_name, grasp_outcomes)
            

    def load_mlp_poses(self):
        # Manually-selected end effector poses
        poses = {
            "block1": [
                (0.0, -0.014147371053695679, 0.17894737422466278, -3.140000104904175, 1.570796251296997, 1.5707963705062866),
                (0.0, -0.0047158002853393555, 0.17894737422466278, -3.140000104904175, 1.570796251296997, 1.5707963705062866),
                (0.0, 0.021221041679382324, 0.17894737422466278, -3.140000104904175, 1.570796251296997, 1.5707963705062866),
                (0.0, 0.0, 0.1894736886024475, -1.5865263938903809, 1.570796251296997, 1.5707963705062866)
            ],
            "block2": [
                (0.0, -0.016505271196365356, 0.17894737422466278, 0.0, 1.570796251296997, 1.5707963705062866),
                (0.0, 0.014147371053695679, 0.17894737422466278, 0.0, 1.570796251296997, 1.5707963705062866),
                (0.0, -0.12261052429676056, 0.10526315867900848, 0.0, 0.4957895278930664, 1.5707963705062866),
                (0.0825263261795044, 0.011789485812187195, 0.13684210181236267, 0.0, 2.3136842250823975, 0.11574292182922363)
            ],
            "block3": [
                (-0.007073685526847839, 0.0, 0.16842105984687805, 0.0, 1.570796251296997, 0.0),
                (0.06837895512580872, 0.0, 0.15263158082962036, 0.09915781021118164, 2.0823161602020264, 0.0),
                (-0.1414736807346344, 0.0070736706256866455, 0.06842105090618134, -0.09915804862976074, 0.4627368450164795, -0.049604058265686035),
                (0.0, -0.10138948261737823, 0.08947368711233139, 3.140000104904175, 0.7602105140686035, 1.5707963705062866),
            ],
            "cylinder1": [
                (0.0, 0.0, 0.17894737422466278, 0.0, 1.570796251296997, 1.5707963705062866),
                (0.0, -0.10138948261737823, 0.1473684161901474, 0.0, 0.8263156414031982, 1.5707963705062866),
                (-0.1178947389125824, 0.0, 0.11578947305679321, 0.0, 0.5618946552276611, -0.049604058265686035),
                (-0.1296842098236084, 0.0, 0.08421052992343903, -3.140000104904175, 0.33052611351013184, -0.033069491386413574)
            ],
            "cylinder2": [
                (0.0, -0.0589473694562912, 0.15789473056793213, 0.0, 1.1898949146270752, 1.5707963705062866),
                (0.0, -0.09903157502412796, 0.12105263024568558, 0.0, 0.7932631969451904, 1.5707963705062866),
                (-0.1061052605509758, -0.016505271196365356, 0.11052631586790085, 0.0, 0.6280002593994141, 0.14881229400634766),
                (0.08016842603683472, 0.009431585669517517, 0.15263158082962036, 0.0, 2.148421049118042, 0.13227760791778564)
            ],
            "cylinder3": [
                (0.0, 0.0, 0.17368420958518982, 0.0, 1.570796251296997, 1.5707963705062866),
                (0.0, -0.10374736785888672, 0.13684210181236267, 0.0, 0.7271578311920166, 1.5707963705062866),
                (-0.10846316814422607, -0.04008421301841736, 0.10526315867900848, 0.0, 0.594947099685669, 0.34722864627838135),
                (-0.10846316814422607, 0.0, 0.11578947305679321, 0.0, 0.7271578311920166, 0.03306937217712402)
            ],
            "bottle1": [
                (0.0, 0.0, 0.24736842513084412, 2.6442105770111084, 1.570796251296997, 1.5707963705062866),
                (0.009431585669517517, -0.13204210996627808, 0.1473684161901474, 0.0, 0.4627368450164795, 1.5707963705062866),
                (0.009431585669517517, -0.12261052429676056, 0.22631579637527466, 0.0, 0.7602105140686035, 1.5707963705062866),
                (0.009431585669517517, -0.12261052429676056, 0.16842105984687805, 0.0, 0.7602105140686035, 1.5707963705062866),
            ],
            "bottle2": [
                (0.009431585669517517, -0.009431570768356323, 0.21578946709632874, 2.3797895908355713, 1.570796251296997, 1.5707963705062866),
                (-0.08016842603683472, 0.04951578378677368, 0.17368420958518982, 0.0, 0.7932631969451904, -0.5952491760253906),
                (-0.09667368233203888, 0.05658945441246033, 0.1473684161901474, 0.0, 0.594947099685669, -0.5291104316711426),
                (-0.08488421142101288, 0.05423158407211304, 0.1473684161901474, -0.033052682876586914, 0.7271578311920166, -0.5787144899368286)
            ],
            "bottle3": [
                (0.0, 0.0, 0.24736842513084412, 2.2806313037872314, 1.570796251296997, 1.5707963705062866),
                (-0.06366315484046936, 0.08724209666252136, 0.16842105984687805, 0.0, 0.5288419723510742, -0.7771308422088623),
                (-0.056589484214782715, 0.09903159737586975, 0.15789473056793213, 0.0, 0.4957895278930664, -0.8928737640380859),
                (0.08016842603683472, -0.07781052589416504, 0.1894736886024475, 0.0, 2.478947401046753, -0.6779226064682007)
            ],
        }
        return poses


    # HELPER FUNCTIONS FOR MLP MODEL
    # Calculate the curvature of a rigid body using the principal curvature estimation algorithm
    # This function only works for meshes that include a .obj file.
    def getRigidBodyCurvature(self, body_id, k):
        mesh_data = p.getMeshData(body_id)
        # Convert the mesh data to a numpy array
        indices, vertices = np.array(mesh_data[0]), np.array(mesh_data[1])

        if indices > 0:
            # Compute the principal curvatures at each vertex
            curvatures = np.zeros((len(vertices), 2))
            tree = cKDTree(vertices)
            for i, vertex in enumerate(vertices):
                # Find the 10 nearest neighbors to the current vertex
                _, indices = tree.query(vertex, k=10)
                neighbors = vertices[indices, :]

                # Compute the covariance matrix of the neighbors
                centroid = np.mean(neighbors, axis=0)
                centered = neighbors - centroid
                cov = np.dot(centered.T, centered)

                # Compute the eigenvalues and eigenvectors of the covariance matrix
                eigenvalues, eigenvectors = np.linalg.eigh(cov)

                # Sort the eigenvalues in descending order
                indices = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[indices]
                eigenvectors = eigenvectors[:, indices]

                # Compute the principal curvatures as the reciprocals of the eigenvalues
                curvatures[i] = np.abs(1.0 / eigenvalues[:2])
            
            # Get the top k eigenvalues
            return curvatures[:k]
        else:
            return np.zeros((k, 2))
    

    # Get geometric features of an object
    def getObjectGeometry(self, body_id):
        curvature_data = self.getRigidBodyCurvature(body_id, k=3).flatten()
        return curvature_data


    # CORE FUNCTIONS FOR RUNNING THE SIMULATION
    def step(self, action, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """

        self.robot.manipulate_ee(action[:-1], control_method)
        self.robot.move_gripper(action[-1])

        # in the step simulation, get the joint orientation
        sixd = self.robot.get_joint_obs()

        self.readResetSimulationButton()
        self.readJointObsButton(sixd)
        self.readBaselineDataCollectionButton()
        self.readMLPDataCollectionButton()
        self.readGetObjectFeaturesButton()
        self.readFetch6dButton(action[:-1])
        
        self.digit_step()
        self.fixed_step_sim(1000)
        
        return self.get_observation()

    def digit_step(self):
        self.color, self.depth = self.digits.render()
        self.digits.updateGUI(self.color, self.depth)


    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs

    def reset(self):
        self.robot.reset()
        return self.get_observation()
    
    # Reset user-defined parameters (GUI slider values)
    def resetUserDebugParameters(self):
        # Re-initialize sliders        
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
        self.gripper_opening_length_control = p.addUserDebugParameter("Gripper opening length", 0, self.robot.gripper_range[1], self.robot.gripper_range[1])
        
        # Re-initialize simulation buttons
        self.initButtonVals()
        self.resetSimulationButton  = p.addUserDebugParameter("Reset simulation", 1, 0, 1)
        self.jointObsButton = p.addUserDebugParameter("Get joint coordinates", 1, 0, 1)
        self.getObjectFeaturesButton = p.addUserDebugParameter("Get object features", 1, 0, 1)
        self.fetch6dButton = p.addUserDebugParameter("Get end effector pose", 1, 0, 1)
        self.baselineDataColButton = p.addUserDebugParameter("Collect data (baseline)", 1, 0, 1)
        self.mlpDataColButton = p.addUserDebugParameter("Collect data (proposed)", 1, 0, 1)

    # Reset the whole simulation
    def reset_simulation(self):
        # Remove sliders and buttons
        p.removeAllUserParameters()

        # Re-initialize sliders and buttons
        self.resetUserDebugParameters()

        self.robot.reset()
        self.container.resetObject()

    def close(self):
        p.disconnect(self.physicsClient)
