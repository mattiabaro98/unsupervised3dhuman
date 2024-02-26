import json

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from skspatial.objects import Plane, Points


class SMPLmeasure:
    """Measure SMPL"""

    def __init__(self):

        with open("./SMPLmeasure/SMPL_index_measure.json") as json_file:
            self.indexes = json.load(json_file)

    def _compute_convex_hull(self, points):
        """
        Compute the convex hull from a 2D numpy array of points.

        Parameters:
            points (numpy.ndarray): Array of shape (n, 2) representing 2D points.

        Returns:
            numpy.ndarray: Array of shape (m, 2) representing the vertices of the convex hull.
        """
        # Ensure the input is a numpy array
        points = np.array(points)

        # Compute the convex hull
        hull = ConvexHull(points)

        # Get the vertices of the convex hull
        hull_vertices = points[hull.vertices]

        return hull_vertices

    def _compute_length(self, xyz):
        points = Points(xyz)
        plane = Plane.best_fit(points)
        normal = np.array(plane.normal)

        y = np.array([0, 1, 0])
        phi = -np.arccos(np.dot(normal, y) / (np.linalg.norm(normal) * np.linalg.norm(y)))
        theta = -np.arctan2(normal[0], normal[2])

        rotX = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
        rotY = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

        for i in range(xyz.shape[0]):
            xyz[i] = np.array(plane.project_point(xyz[i]))

        for i in range(xyz.shape[0]):
            xyz[i] = np.dot(rotX, np.dot(rotY, xyz[i]))

        xz = xyz[:, [0, 2]]

        convex_hull = self._compute_convex_hull(xz)

        length = 0

        for i in range(1, convex_hull.shape[0]):
            length += np.linalg.norm(convex_hull[i] - convex_hull[i - 1])

        length += np.linalg.norm(convex_hull[-1] - convex_hull[0])

        return length

    def measure_smpl(self, filename, height):

        mesh = trimesh.load(filename)
        xyz = mesh.vertices

        real_height = height
        upper_y = np.mean(xyz[self.indexes["head_tip"]][:, 1])
        lower_y = np.mean(xyz[self.indexes["feet_soles"]][:, 1])
        smpl_height = upper_y - lower_y

        left_tight = self._compute_length(xyz[self.indexes["left_tight"]]) * real_height / smpl_height
        right_tight = self._compute_length(xyz[self.indexes["right_tight"]]) * real_height / smpl_height
        left_calf = self._compute_length(xyz[self.indexes["left_calf"]]) * real_height / smpl_height
        right_calf = self._compute_length(xyz[self.indexes["right_calf"]]) * real_height / smpl_height
        chest = self._compute_length(xyz[self.indexes["chest"]]) * real_height / smpl_height
        waist = self._compute_length(xyz[self.indexes["waist"]]) * real_height / smpl_height
        hip = self._compute_length(xyz[self.indexes["hip"]]) * real_height / smpl_height
        left_arm = self._compute_length(xyz[self.indexes["left_arm"]]) * real_height / smpl_height
        right_arm = self._compute_length(xyz[self.indexes["right_arm"]]) * real_height / smpl_height

        results = {
            "left_tight": left_tight,
            "right_tigh": right_tight,
            "left_calf ": left_calf,
            "right_calf": right_calf,
            "chest": chest,
            "waist": waist,
            "hip": hip,
            "left_arm": left_arm,
            "right_arm": right_arm,
        }

        return results
