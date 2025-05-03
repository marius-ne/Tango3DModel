import trimesh
import pyrender
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import json
import os
import open3d as o3d

class ModelRenderer:
    def __init__(self, filename):
        # Load mesh using trimesh
        self.mesh = trimesh.load(filename)
        if not isinstance(self.mesh, trimesh.Trimesh):
            self.mesh = self.mesh.dump().sum()  # Combine into single mesh if needed

        # Create pyrender scene and add mesh
        self.scene = pyrender.Scene()
        mesh = pyrender.Mesh.from_trimesh(self.mesh, smooth=True)
        self.scene.add(mesh)

        self.o3dmesh = o3d.io.read_triangle_mesh(filename)
        

    def render(self, 
               intrinsic_matrix: np.ndarray, 
               extrinsic_matrix: np.ndarray):
        
        # Create camera from intrinsics
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

        width = int(intrinsic_matrix[0,2]*2)
        height = int(intrinsic_matrix[1,2]*2)
        # Add camera to scene
        cam_node = self.scene.add(camera, pose=extrinsic_matrix)

        # Use OffscreenRenderer
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        color, _ = renderer.render(self.scene)

        # Clean up
        self.scene.remove_node(cam_node)
        renderer.delete()

        return color
    
    def generate_random_positions(self):
        if self.strategy == "uniform_cartesian":
            distances = np.random.uniform(self.min_distance, self.max_distance, size=(self.num_candidates))
        elif self.strategy == "uniform_spherical":
            distances = np.random.uniform(self.min_distance, self.max_distance, size=(self.num_candidates))
            distances = np.sqrt(distances) # distribute evenly in a sphere
        positions = np.column_stack((distances, np.zeros((self.num_candidates, 2))))
        return positions

    def generate_random_attitudes(self):
        attitudes = np.zeros((self.num_candidates, 4))
        for i in range(self.num_candidates):
            attitudes[i, :] = Rotation.random().as_quat()
        return attitudes

    def generate_random_camera_positions(self):

        attitudes = self.generate_random_attitudes()
        positions = self.generate_random_positions()
        camera_positions = np.zeros((self.num_candidates, 3))
        for i in range(self.num_candidates):
            camera_positions[i, :] = Rotation.from_quat(attitudes[i]).apply(positions[i])
        return camera_positions
    
    def get_active_camera_pose_for_camera_position(self, camera_position_world):
        """
        Gives active transformation (translates points in the world coordinate frame).
        Maps origin to camera center.
        """
        # Handle the case where the camera is at the origin
        if np.linalg.norm(camera_position_world) < 1e-6:
            raise ValueError("Warning: Camera is at the origin, projection is potentially unstable.")
    
        camera_to_origin_world = -camera_position_world
        camera_to_origin_world /= np.linalg.norm(camera_to_origin_world)  # Normalize

        desired_camera_axis = camera_to_origin_world  # Camera Z-axis (looking towards the origin)
        current_camera_axis = np.array([0, 0, 1])  # Default camera Z-axis

        rotation, _ = Rotation.align_vectors([desired_camera_axis], [current_camera_axis])

        r_vec = rotation.as_rotvec()

        return r_vec, camera_to_origin_world
    
    def get_passive_camera_pose_for_camera_position(self, camera_position_world):
        """
        Gives passive transformation (changes reference frames).
        Maps origin to a point on the boresight axis.
        """
        # Handle the case where the camera is at the origin
        if np.linalg.norm(camera_position_world) < 1e-6:
            raise ValueError("Warning: Camera is at the origin, projection is potentially unstable.")
    
        r_vec, t_vec = self.get_active_camera_pose_for_camera_position(camera_position_world)
        # Invert the pose to get the passive camera pose
        passive_rvec, passive_tvec = self.invert_pose(r_vec, t_vec)

        return passive_rvec, passive_tvec
    
    def generate_random_camera_poses(self):
        positions = self.generate_random_camera_positions()
        camera_poses = np.zeros((self.num_candidates, 6))
        for i in range(self.num_candidates):
            rvec, tvec = self.get_active_camera_pose_for_camera_position(positions[i])
            
            camera_poses[i] = np.hstack((rvec, tvec))

        return camera_poses
    
    def invert_pose(self, rvec, tvec):
        """
        Inverts the camera pose (rotation and translation).
        """
        # Invert the rotation vector
        rotation = Rotation.from_rotvec(rvec).inv()
        inverted_rvec = rotation.as_rotvec()
        
        # Invert the translation vector
        inverted_tvec = -rotation.apply(tvec)

        return inverted_rvec, inverted_tvec
    
    def visualize_camera_positions(self, camera_positions, colors=None):
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(camera_positions))

        # Handle colors
        if colors is None:
            colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(camera_positions), 1))  # red
        else:
            colors = np.array(colors) / 255.0  # Normalize to [0, 1]

        pcd.colors = o3d.utility.Vector3dVector(colors)

        geometry_list = [pcd]
        geometry_list.append(self.o3dmesh)

        # Visualize
        o3d.visualization.draw_geometries(geometry_list)

    def draw_pose_axes_on_image(self, image, K, rvec, tvec):
        # Create a 3D point cloud for the axes
        axes_length = 1
        axes_points = np.array([[0, 0, 0], [axes_length, 0, 0], [0, axes_length, 0], [0, 0, axes_length]])
        
        axes_on_image = cv2.projectPoints(axes_points.astype(np.float32), rvec, tvec, K, None)
        axes_on_image = axes_on_image[0].squeeze()
        print(axes_on_image)
        output_image = image.copy()
        for i in range(3):
            start_point = tuple(axes_on_image[0].astype(int))
            end_point = tuple(axes_on_image[i + 1].astype(int))
            color = (255, 0, 0) if i == 0 else (0, 255, 0) if i == 1 else (0, 0, 255)
            cv2.line(output_image, start_point, end_point, color, 10)
        return output_image


    
if __name__ == "__main__":
    speed_path = "E:\ESA\VBN_DataSets\Data\SPEED+\synthetic\images"
    model_path = "models/obj/Tango_rotated.obj"
    renderer = ModelRenderer(model_path)
    renderer.num_candidates = 10
    renderer.strategy = "uniform_spherical"
    renderer.min_distance = 2
    renderer.max_distance = 5

    poses = renderer.generate_random_camera_poses()

    with open("data/train.json", "r") as f:
        speed_poses = json.load(f)
        n_poses = len(speed_poses)

    K = np.array([[
      2988.5795163815555,
      0,
      960
    ],
    [
      0,
      2988.3401159176124,
      600
    ],
    [
      0,
      0,
      1
    ]])

    # Convert K to Open3D format
    intrinsic_matrix = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])

    
    for pose in speed_poses:
        i = np.random.randint(0, n_poses)
        pose = speed_poses[i]

        print(pose['filename'])
        # Renderer wants CAMERA -> WORLD (passive, so in world coordinates)
        # Speed gives WORLD -> CAMERA (passive)
        speed_t_vec = np.array(pose["r_Vo2To_vbs_true"])
        # For some weird reason the position is inverted in the speed dataset
        #speed_t_vec = -speed_t_vec  # Invert the translation vector

        attitude = pose["q_vbs2tango_true"]
        changed_attitude = np.array([attitude[1], attitude[2], attitude[3], attitude[0]]) 
        #attitude = np.array([attitude[3], attitude[0], attitude[1], attitude[2]]) 
        speed_r_vec = Rotation.from_quat(changed_attitude).as_rotvec()  # Rotation vector
        
        r_vec, t_vec = speed_r_vec, speed_t_vec
        
        r_vec_inverted, t_vec_inverted = renderer.invert_pose(speed_r_vec, -speed_t_vec)

        #renderer.visualize_camera_positions([t_vec])

        r_mat = Rotation.from_rotvec(r_vec_inverted).as_matrix()

        extrinsic_matrix = np.hstack((r_mat, t_vec_inverted.reshape(3, 1)))
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))

        image = renderer.render(K, extrinsic_matrix)
        #cv2.imshow("Rendered",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        speed_image_path = os.path.join(speed_path, pose['filename'])
        speed_image = cv2.imread(speed_image_path)

        # Create mask for the rendered image (just invert)
        rendered_mask = cv2.bitwise_not(image[:, :, 0])
        
        # Create mask for the speed image
        speed_mask = np.zeros((speed_image.shape[0], speed_image.shape[1]), dtype=np.uint8)
        speed_mask[speed_image[:, :, 0] > 10] = 255

        image_axes = renderer.draw_pose_axes_on_image(image, K, speed_r_vec, speed_t_vec)
        speed_image_axes = renderer.draw_pose_axes_on_image(speed_image, K, speed_r_vec, speed_t_vec)
        
        # Check if 50% of the mask is black (otherwise probably has background)
        if np.sum(speed_mask == 0) / speed_mask.size < 0.5:
            print("Image has background, skipping")
            continue

        cv2.imshow("Rendered Mask", rendered_mask)
        cv2.imshow("Speed Mask", speed_mask)

        # Compute IoU
        intersection = cv2.bitwise_and(rendered_mask, speed_mask)
        union = cv2.bitwise_or(rendered_mask, speed_mask)
        iou = np.sum(intersection) / np.sum(union)
        print(f"IoU: {iou}")
            
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #break
