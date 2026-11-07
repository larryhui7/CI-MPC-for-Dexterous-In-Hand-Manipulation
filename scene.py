"""Scene: Tesollo DG-3F-M welded to the world + a sphere on a vertical revolute joint. ROTZ

Generated model files (written to <build_dir>/models, never hand-edited):
  hand_planning.urdf  official kinematics/inertias, collision = spheres, no visuals
  hand_visual.urdf    same, plus the official .dae visual meshes (Meshcat only)
  object.sdf          sphere on an unactuated revolute joint about world z
  directives.yaml     qsim model directive: weld hand mount to world

Why the collision geometry is replaced (see README for details):
  * Drake 1.58 cannot use the official .STL collision meshes at all.
  * qsim's A = df/dx uses AutoDiff signed distances, which Drake supports only
    for Sphere-{Sphere, Box, HalfSpace}. Capsules/cylinders throw.
  => every hand collision shape is a sphere and the object is a sphere.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, MeshcatVisualizer,
    MeshcatVisualizerParams, Parser, QueryObject, Rgba, RigidTransform, Role, Sphere,
)

FINGERS = (1, 2, 3)
HAND_JOINTS = [f"j_dg_{f}_{k}" for f in FINGERS for k in (1, 2, 3, 4)]
TIP_FRAMES = [f"l_dg_{f}_tip" for f in FINGERS]
OBJECT_JOINT = "object_yaw"
DRAKE_NS = "http://drake.mit.edu"
NX, NU = 13, 12  # RotZ: x = [12 hand joints, object yaw]; u = hand joint increments
# Meshcat contact dots, one colour per finger (magenta, orange, yellow).
CONTACT_COLORS = {1: Rgba(0.95, 0.15, 0.60, 1.0), 2: Rgba(1.00, 0.55, 0.00, 1.0),
                  3: Rgba(1.00, 0.85, 0.00, 1.0)}


def is_free(cfg):
    """Free system: the object is a free body held only by the fingers (no
    support), q_o = [quaternion wxyz; position xyz]. Otherwise RotZ (Fig. 4a): q_o = yaw."""
    return cfg.get("system", "rotz") == "free"


def n_q_object(cfg):
    return 7 if is_free(cfg) else 1


def n_v_object(cfg):
    return 6 if is_free(cfg) else 1


def object_center(cfg, x):
    """World position of the object center for state x."""
    if is_free(cfg):
        return np.asarray(x[NU + 4:NU + 7], float)
    return np.asarray(cfg["obj_center"], float)


def finger_sign(link):
    """Finger 1 links extend along +x of their frames, fingers 2/3 along -x."""
    return 1.0 if link.startswith("l_dg_1_") else -1.0


def write_model_files(cfg):
    out = os.path.abspath(os.path.join(cfg["build_dir"], "models"))
    os.makedirs(out, exist_ok=True)
    files = {"dir": out}
    for variant in ("planning", "visual"):
        path = os.path.join(out, f"hand_{variant}.urdf")
        _write_hand_urdf(cfg, path, keep_visuals=(variant == "visual"))
        files[f"hand_{variant}"] = path
    files["object"] = os.path.join(out, "object.sdf")
    _write_object_sdf(cfg, files["object"])
    files["directives"] = os.path.join(out, "directives.yaml")
    with open(files["directives"], "w") as fh:
        fh.write(
            "directives:\n"
            f"- add_model:\n    name: hand\n    file: file://{files['hand_planning']}\n"
            "- add_weld:\n    parent: world\n    child: hand::l_dg_mount\n")
    return files


def _write_hand_urdf(cfg, path, keep_visuals):
    ET.register_namespace("drake", DRAKE_NS)
    tree = ET.parse(cfg["urdf"])
    robot = tree.getroot()
    mesh_dir = os.path.abspath(cfg["mesh_dir"])
    sphere_specs = cfg["hand_spheres"]
    for link in robot.findall("link"):
        name = link.get("name")
        for col in link.findall("collision"):
            link.remove(col)
        for vis in link.findall("visual"):
            if not keep_visuals:
                link.remove(vis)
                continue
            mesh = vis.find("geometry/mesh")
            mesh.set("filename", "file://" + os.path.join(
                mesh_dir, os.path.basename(mesh.get("filename"))))
        suffix = name.split("_")[-1]  # l_dg_2_tip -> "tip", l_dg_2_4 -> "4"
        if name.startswith("l_dg_") and name[5] in "123" and suffix in sphere_specs:
            for i, spec in enumerate(sphere_specs[suffix]):
                dist, z, radius = spec if len(spec) == 3 else (spec[0], 0.0, spec[1])
                col = ET.SubElement(link, "collision", name=f"{name}_sphere{i}")
                ET.SubElement(col, "origin", xyz=f"{finger_sign(name) * dist} 0 {z}",
                              rpy="0 0 0")
                ET.SubElement(ET.SubElement(col, "geometry"), "sphere",
                              radius=str(radius))
                props = ET.SubElement(col, f"{{{DRAKE_NS}}}proximity_properties")
                for tag in ("mu_static", "mu_dynamic"):
                    ET.SubElement(props, f"{{{DRAKE_NS}}}{tag}", value=str(cfg["mu"]))
    # Contacts in Eq. (2): hand-object and finger-finger (fingers cannot pass
    # through each other). Links of the same finger are filtered: their spheres
    # overlap by construction at the joints.
    for f in FINGERS:
        group = ET.SubElement(robot, f"{{{DRAKE_NS}}}collision_filter_group",
                              name=f"finger_{f}")
        for link in robot.findall("link"):
            if link.get("name").startswith(f"l_dg_{f}_"):
                ET.SubElement(group, f"{{{DRAKE_NS}}}member", link=link.get("name"))
        ET.SubElement(group, f"{{{DRAKE_NS}}}ignored_collision_filter_group",
                      name=f"finger_{f}")
    tree.write(path)


def _write_object_sdf(cfg, path):
    r, (cx, cy, cz), m = cfg["obj_radius"], cfg["obj_center"], cfg["obj_mass"]
    inertia = 0.4 * m * r * r
    joint = "" if is_free(cfg) else f"""
    <!-- Unactuated: effort 0 => Drake attaches no actuator to this joint. -->
    <joint name="{OBJECT_JOINT}" type="revolute">
      <parent>world</parent>
      <child>object</child>
      <axis><xyz>0 0 1</xyz><limit><effort>0</effort></limit></axis>
    </joint>"""  # Free: no joint, so Drake makes the object a free (6-dof) body
    with open(path, "w") as fh:
        fh.write(f"""<?xml version="1.0"?>
<sdf version="1.7" xmlns:drake="{DRAKE_NS}">
  <model name="object">
    <link name="object">
      <pose>{cx} {cy} {cz} 0 0 0</pose>
      <inertial>
        <mass>{m}</mass>
        <inertia><ixx>{inertia}</ixx><iyy>{inertia}</iyy><izz>{inertia}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <collision name="object_sphere">
        <geometry><sphere><radius>{r}</radius></sphere></geometry>
        <drake:proximity_properties>
          <drake:mu_static>{cfg['mu']}</drake:mu_static>
          <drake:mu_dynamic>{cfg['mu']}</drake:mu_dynamic>
        </drake:proximity_properties>
      </collision>
      <visual name="object_visual">
        <geometry><sphere><radius>{r}</radius></sphere></geometry>
        <material><diffuse>0.2 0.4 0.9 0.6</diffuse></material>
      </visual>
    </link>{joint}
  </model>
</sdf>
""")


def build_plant(files, visual=False, meshcat=None):
    """pydrake plant of the same scene (used for inspection, IK, visualization).

    The CQDC model lives in qsim's own C++ plant (ccp.py); this one is built
    from the same generated files so kinematics and geometry are identical.
    """
    builder = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(builder, 0.0)
    parser = Parser(plant)
    obj = parser.AddModels(files["object"])[0]
    hand = parser.AddModels(files["hand_visual" if visual else "hand_planning"])[0]
    plant.RenameModelInstance(hand, "hand")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("l_dg_mount", hand))
    plant.Finalize()
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, sg, meshcat, MeshcatVisualizerParams(
            role=Role.kIllustration, prefix="visual"))
        MeshcatVisualizer.AddToBuilder(builder, sg, meshcat, MeshcatVisualizerParams(
            role=Role.kProximity, prefix="collision",
            default_color=Rgba(0.1, 0.8, 0.2, 0.5), visible_by_default=True))
    diagram = builder.Build()
    s = dict(diagram=diagram, plant=plant, sg=sg, hand=hand, obj=obj, meshcat=meshcat)
    s["ctx"] = diagram.CreateDefaultContext()
    s["pctx"] = plant.GetMyMutableContextFromRoot(s["ctx"])
    # Map our state x = [hand joints (HAND_JOINTS order), q_o] -> plant q.
    if plant.HasJointNamed(OBJECT_JOINT, obj):  # RotZ: q_o = yaw
        obj_q = [plant.GetJointByName(OBJECT_JOINT, obj).position_start()]
    else:  # Free: q_o = [quaternion wxyz; position xyz]
        start = plant.GetBodyByName("object", obj).floating_positions_start()
        obj_q = list(range(start, start + 7))
    s["x_to_q"] = np.array(
        [plant.GetJointByName(j, hand).position_start() for j in HAND_JOINTS] + obj_q)
    return s


def set_x(s, x):
    q = np.zeros(s["plant"].num_positions())
    q[s["x_to_q"]] = x
    s["plant"].SetPositions(s["pctx"], q)


def joint_limits(s):
    p, hand = s["plant"], s["hand"]
    q_lo = np.array([p.GetJointByName(j, hand).position_lower_limits()[0] for j in HAND_JOINTS])
    q_hi = np.array([p.GetJointByName(j, hand).position_upper_limits()[0] for j in HAND_JOINTS])
    return q_lo, q_hi


def signed_distances(s, x):
    """All unfiltered hand-object pairs (finger-finger pairs left out): list of
    (hand_body, phi [m])."""
    set_x(s, x)
    qo: QueryObject = s["sg"].get_query_output_port().Eval(
        s["sg"].GetMyContextFromRoot(s["ctx"]))
    insp = qo.inspector()
    out = []
    for sdp in qo.ComputeSignedDistancePairwiseClosestPoints(1.0):
        names = [s["plant"].GetBodyFromFrameId(insp.GetFrameId(g)).name()
                 for g in (sdp.id_A, sdp.id_B)]
        if "object" not in names:
            continue
        hand_body = names[0] if names[1] == "object" else names[1]
        out.append((hand_body, sdp.distance))
    return out


def tip_sphere_center(s, finger, cfg):
    dist = cfg["hand_spheres"]["tip"][0][0]
    frame = s["plant"].GetFrameByName(f"l_dg_{finger}_tip", s["hand"])
    return s["plant"].CalcPointsPositions(
        s["pctx"], frame, [finger_sign(frame.name()) * dist, 0, 0],
        s["plant"].world_frame()).ravel()


def tip_contacts(s, cfg, x):
    """Per fingertip: signed distance tip sphere <-> object [m] and the closest
    point on the object surface (world frame). Sphere-sphere geometry, the same
    pair qsim uses."""
    set_x(s, x)
    c, r = object_center(cfg, x), cfg["obj_radius"]
    r_tip = cfg["hand_spheres"]["tip"][0][1]
    phi, points = np.zeros(len(FINGERS)), np.zeros((len(FINGERS), 3))
    for j, f in enumerate(FINGERS):
        d = tip_sphere_center(s, f, cfg) - c
        dist = np.linalg.norm(d)
        phi[j] = dist - r - r_tip
        points[j] = c + r * d / dist
    return phi, points


def _add_contact_dots(s, cfg):
    for f in FINGERS:
        s["meshcat"].SetObject(f"contacts/tip{f}", Sphere(cfg["contact_dot_radius"]),
                               CONTACT_COLORS[f])


def _set_contact_dots(s, cfg, x, time=None):
    """Move each finger's dot to its contact point; show it only while the tip
    is within cfg["contact_phi"]. With `time`, the change goes into the recording."""
    phi, points = tip_contacts(s, cfg, x)
    for j, f in enumerate(FINGERS):
        path = f"contacts/tip{f}"
        s["meshcat"].SetTransform(path, RigidTransform(points[j]), time_in_recording=time)
        s["meshcat"].SetProperty(path, "visible", bool(phi[j] < cfg["contact_phi"]),
                                 time_in_recording=time)


def initial_grasp(s, cfg):
    """Symmetric grasp: each finger plane passes through the object axis and its
    tip sphere sits `grasp_gap` outside the object surface at elevation
    `grasp_elevation`. j1 aligns each finger radially, j2 = 0, (j3, j4) by IK."""
    r_target = cfg["obj_radius"] + cfg["hand_spheres"]["tip"][0][1] + cfg["grasp_gap"]
    c = np.asarray(cfg["obj_center"])
    el = cfg["grasp_elevation"]
    x = np.zeros(NU + n_q_object(cfg))
    if is_free(cfg):
        x[NU:] = [1.0, 0.0, 0.0, 0.0, *c]  # identity orientation at obj_center
    for i, (f, j1, azim) in enumerate(zip(FINGERS, cfg["grasp_j1"], cfg["finger_azimuth"])):
        target = c + r_target * np.array(
            [np.cos(azim) * np.cos(el), np.sin(azim) * np.cos(el), np.sin(el)])

        def residual(j34):
            x[4 * i:4 * i + 4] = [j1, 0.0, j34[0], j34[1]]
            set_x(s, x)
            return tip_sphere_center(s, f, cfg) - target

        sol = least_squares(residual, cfg["grasp_ik_guess"], bounds=([0, -1.57], [2.81, 1.86]),
                            xtol=1e-14, ftol=1e-14, gtol=1e-14)
        assert np.linalg.norm(sol.fun) < 1e-6, f"grasp IK failed for finger {f}: {sol.fun}"
        x[4 * i:4 * i + 4] = [j1, 0.0, *sol.x]
    return x


def mesh_clearance(s, cfg, x):
    """Distance from the object sphere to the official (convex hulls of) hand
    collision meshes, vertex-based: a check that the simplified spheres do not
    hide a real penetration of the actual DG-3F-M geometry."""
    set_x(s, x)
    c, r = np.asarray(cfg["obj_center"]), cfg["obj_radius"]
    out = {}
    for f in FINGERS:
        for part, mesh in (("3", "3"), ("4", "4"), ("tip", "tip")):
            link = f"l_dg_{f}_{part}"
            stl = os.path.join(cfg["mesh_dir"], f"l_{'1' if f == 1 else '23'}_{mesh}_c.STL")
            v = _load_stl(stl)
            X = s["plant"].EvalBodyPoseInWorld(
                s["pctx"], s["plant"].GetBodyByName(link, s["hand"]))
            out[link] = np.min(np.linalg.norm((X.multiply(v.T)).T - c, axis=1)) - r
    return out


def _load_stl(path):
    data = open(path, "rb").read()
    n = int.from_bytes(data[80:84], "little")
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype(
        [("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3).astype(float)


def show(s, x, cfg=None):
    """Publish state x; with cfg, also draw the fingertip contact dots."""
    set_x(s, x)
    s["diagram"].ForcedPublish(s["ctx"])
    if cfg is not None:
        _add_contact_dots(s, cfg)
        _set_contact_dots(s, cfg, x)


def animate(s, xs, h, cfg):
    """Record states xs (one per time step h) as a Meshcat animation, with a
    coloured dot at each fingertip contact (tip within cfg["contact_phi"])."""
    show(s, xs[0], cfg)  # live scene = first frame (also creates the dots)
    vis = [v for v in s["diagram"].GetSystems() if isinstance(v, MeshcatVisualizer)]
    for v in vis:
        v.StartRecording(set_transforms_while_recording=False)
    for i, x in enumerate(xs):
        s["ctx"].SetTime(i * h)
        show(s, x)
        _set_contact_dots(s, cfg, x, time=i * h)
    for v in vis:
        v.StopRecording()
        v.PublishRecording()
