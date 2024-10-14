import polyscope as ps
from symmetries.symmetries import SymmetryPlane
from symmetries.symmetries import SymmetryAxis
import numpy as np
class Screenshots():
    def __init__(self, save_path, enabled=False, extension=".jpg", object_id=""):
        self.enabled = enabled
        self.extension = extension
        self.save_path = save_path
        self.object_id = object_id

    def object_and_planar_syms(self, object_numpy_only_one, normals_numpy_list, transforms_numpy_list=None, object_filename=None, sym_plane_colors=((1, 0.5, 0.25),
                                                 (1, 0.5, 0.5),
                                                 (1, 0.25, 0.15)), correct_id=None):
        pass

    def object_and_rot_syms(self, object_numpy_only_one, normals_numpy_list, transforms_numpy_list=None, object_filename=None,
                           sym_axis_colors=((1, 0.5, 0.25),
                                                 (1, 0.5, 0.5),
                                                 (1, 0.25, 0.15))):
        pass
    def objects_multiple(self, object_numpy_list):
        pass

    def reset_renderer(self):
        pass
class ScreenshotsPolyscope(Screenshots):
    def __init__(self, save_path, enabled=False, extension="", object_id=""):
        super().__init__(save_path, enabled, extension)
        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps.look_at((1.5, 1.5, 1.5), (0., 0., 0.))
        ps.set_shadow_darkness(0.4)

    def reset_renderer(self):
        ps.remove_all_structures()

    def object_and_rot_syms(self, object_numpy_only_one, normals_numpy_list, transforms_numpy_list=None, object_filename=None,
                           sym_axis_colors=((1, 0.5, 0.25),
                                                 (1, 0.5, 0.5),
                                                 (1, 0.25, 0.15))):

        transforms_per_obj = 3
        angle_rots = 3
        if object_filename is None:
            object_filename = self.object_id

        sym_objs = [SymmetryAxis(np.array([0, 0, 0]), n,  float("inf")) for n in normals_numpy_list]
        object_register = ps.register_point_cloud("object_point_cloud", object_numpy_only_one, color=(1, 0, 0),
                                                  enabled=True)

        transform_clouds = []
        if transforms_numpy_list is not None:
            colors = [(0, 0.1, 1),
                      (0.1, 0.2, 1),
                      (0.2, 0.1, 1)]
            for i, transform_cloud in enumerate(transforms_numpy_list):
                transform_cloud = ps.register_point_cloud(f"transform_cloud{i}", transform_cloud, color=colors[i%3], enabled=False)
                transform_clouds.append(transform_cloud)


        register_axis = []
        register_axis_onlyonedirection = []
        for i, sym in enumerate(sym_objs):
            register_axis.append(ps.register_curve_network(f"sym_axis_{i}", np.array([0 * sym.normal + sym.point, 0.8 * sym.normal + sym.point]),
                                  np.array([[0, 1]]), radius=0.01, enabled=False, color=sym_axis_colors[i], transparency=0.8))
            register_axis_onlyonedirection.append(ps.register_curve_network(f"sym_axis_{i}2", np.array([0 * sym.normal + sym.point, 0.8 * sym.normal + sym.point]),
                                  np.array([[0, 1]]), radius=0.01, enabled=False, color=sym_axis_colors[i]))

        #illustrative rotations separate only one
        for i in range(transforms_per_obj):
            transform_clouds[i].set_enabled(True)
            ps.screenshot(str(self.save_path / f"{object_filename}_only_rotated_by_angle_n_{i}.jpg"))
            transform_clouds[i].set_enabled(False)

        #take photos
        for i, reg_axis in enumerate(register_axis):
            reg_axis.set_enabled(True)
            ps.screenshot(str(self.save_path / f"{object_filename}_sym_axis_{i}.jpg"))

            #with transforms
            for j in range(transforms_per_obj):
                transform_clouds[i*transforms_per_obj+j].set_enabled(True)
            ps.screenshot(str(self.save_path / f"{object_filename}_with_transforms_sym_axis_{i}.jpg"))
            for j in range(transforms_per_obj):
                transform_clouds[i*transforms_per_obj+j].set_enabled(False)

            reg_axis.set_enabled(False)

        #all axis only
        for i, reg_axis in enumerate(register_axis):
            reg_axis.set_enabled(True)
        ps.screenshot(str(self.save_path / f"{object_filename}_all_sym_axis_.jpg"))
        for i, reg_axis in enumerate(register_axis):
            reg_axis.set_enabled(False)


        object_register.set_enabled(False)
        self.reset_renderer()

    def object_and_planar_syms(self, object_numpy_only_one, normals_numpy_list, transforms_numpy_list=None, object_filename=None,
                               sym_plane_colors=((1, 0.5, 0.25),
                                                 (1, 0.5, 0.5),
                                                 (1, 0.25, 0.15)), correct_id=None):


        if object_filename is None:
            object_filename = self.object_id
        sym_objs = [SymmetryPlane(np.array([0,0,0]), n) for n in normals_numpy_list]
        normal_objs = [SymmetryAxis(np.array([0, 0, 0]), n,  float("inf")) for n in normals_numpy_list]

        register_object = ps.register_point_cloud("object_point_cloud", object_numpy_only_one, color=(0.11,0.56, 1), enabled=True)

        transform_clouds = []
        normal_obj_register = []
        for i, sym in enumerate(normal_objs):
            normal_obj_register.append(ps.register_curve_network(f"sym_axis_{i}", np.array([0 * sym.normal + sym.point, 1.2 * sym.normal + sym.point]),
                                  np.array([[0, 1]]), radius=0.01, enabled=False, color=(0, 0, 0)))

        if transforms_numpy_list is not None:
            colors = [(0, 0.1, 1),
                      (0.1, 0.2, 1),
                      (0.2, 0.1, 1)]

            for i, transform_cloud in enumerate(transforms_numpy_list):
                transform_cloud = ps.register_point_cloud(f"transform_cloud{i}", transform_cloud, color=colors[i], enabled=True)
                transform_clouds.append(transform_cloud)
                normal_obj_register[i].set_enabled(True)
                ps.screenshot(str(self.save_path / f"{object_filename}_transform{i}.jpg"))
                normal_obj_register[i].set_enabled(False)
                transform_cloud.set_enabled(False)





        mesh_register = []
        for i, sym in enumerate(sym_objs):

            mesh = ps.register_surface_mesh("sym_plane_" + str(i), sym.coords, sym.trianglesBase, color=sym_plane_colors[i], enabled=True)
            mesh_register.append(mesh)
            mesh.set_transparency(0.8)

            if i == correct_id.item():
                sym_plane_txt = str(self.save_path / f"{object_filename}_sym_plane_{i}_correct.jpg")
            else:
                sym_plane_txt = str(self.save_path / f"{object_filename}_sym_plane_{i}.jpg")
            ps.screenshot(sym_plane_txt)
            if transforms_numpy_list is not None:
                transform_clouds[i].set_enabled(True)
                ps.screenshot(str(self.save_path / f"{object_filename}_sym_plane_with_transform{i}.jpg"))
                transform_clouds[i].set_enabled(False)
            mesh.set_enabled(False)

        #all at once
        for i, mesh in enumerate(mesh_register):
            mesh.set_enabled(True)

        ps.screenshot(str(self.save_path / f"{object_filename}_all_sym_planes.jpg"))

        ps.show()

        for i, mesh in enumerate(mesh_register):
            mesh.set_enabled(False)
        self.reset_renderer()

class NullScreenshot(Screenshots):
    def __init__(self):
        pass


class ScreenshotsMitsuba(Screenshots):
    def __init__(self, save_path, enabled=False, extension="", object_id=""):
        super().__init__(save_path, enabled, extension)
        pass

    def reset_renderer(self):
        pass
    def object_and_rot_syms(self, object_numpy_only_one, normals_numpy_list, transforms_numpy_list=None,
                            object_filename=None,
                            sym_axis_colors=((1, 0.5, 0.25),
                                             (1, 0.5, 0.5),
                                             (1, 0.25, 0.15))):
        pass

    def object_and_planar_syms(self, object_numpy_only_one, normals_numpy_list, transforms_numpy_list=None, object_filename=None,
                                   sym_plane_colors=((1, 0.5, 0.25),
                                                     (1, 0.5, 0.5),
                                                     (1, 0.25, 0.15))):
        pass




