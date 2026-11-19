"""Scene: a hand (Tesollo DG-3F-M or Allegro, see HANDS) welded to the world + a
sphere on a vertical revolute joint (RotZ) or free (Free).

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
    MeshcatVisualizerParams, Parser, QueryObject, Rgba, RigidTransform, Role,
    RollPitchYaw, Sphere,
)

OBJECT_JOINT = "object_yaw"
DRAKE_NS = "http://drake.mit.edu"
_URDFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "URDFs")


def _finger(joints, links, parts, axis, side):
    """One finger: its 4 joints (base to tip), its links, the key of each link in
    cfg["hand_spheres"] (None = no collision spheres), and in the link frames
    the direction the finger extends along (axis) and a sideways offset
    direction (side). The last link is the fingertip."""
    return dict(joints=joints, links=links, parts=parts, axis=np.array(axis, float),
                side=np.array(side, float))


# Hand models. root = link welded to the world with roll-pitch-yaw weld_rpy_deg;
# squeeze / tangential = per finger, the joint (0-3) that closes the finger onto
# the object / moves its tip sideways (used by the run_rotz step and grad tests).
HANDS = {
    "dg3fm": dict(  # Tesollo DG-3F-M: 3 fingers x 4 joints, base down, fingers up
        urdf=os.path.join(_URDFS, "dg3fm", "dg3f_m.urdf"),
        mesh_dir=os.path.join(_URDFS, "dg3fm", "meshes"),
        root="l_dg_mount", weld_rpy_deg=[0.0, 0.0, 0.0],
        # finger 1 links extend along +x of their frames, fingers 2/3 along -x
        fingers=[_finger([f"j_dg_{f}_{k}" for k in (1, 2, 3, 4)],
                         [f"l_dg_{f}_{k}" for k in (1, 2, 3, 4, "tip")],
                         ["1", "2", "3", "4", "tip"],
                         [1.0 if f == 1 else -1.0, 0, 0], [0, 0, 1]) for f in (1, 2, 3)],
        squeeze=[2, 2, 2], tangential=[0, 0, 0],
        # official collision meshes, for scene.mesh_clearance: (link, STL file)
        clearance_meshes=[(f"l_dg_{f}_{k}", f"l_{'1' if f == 1 else '23'}_{k}_c.STL")
                          for f in (1, 2, 3) for k in ("3", "4", "tip")],
    ),
    "allegro": dict(  # Allegro right hand: index, middle, ring, thumb x 4 joints
        urdf=os.path.join(_URDFS, "allegro", "allegro_hand_description_right.urdf"),
        mesh_dir=os.path.join(_URDFS, "allegro", "meshes"),
        # palm up: palm normal (hand +x) -> world +z, fingers (hand +z) -> world -x
        root="hand_root", weld_rpy_deg=[0.0, -90.0, 0.0],
        # every link extends along +z of its frame; +x is the palm side
        fingers=[_finger([f"joint_{4 * f + k}" for k in range(4)],
                         [f"link_{4 * f + k}" for k in range(4)] + [f"link_{4 * f + 3}_tip"],
                         [None, "1", "2", None, "tip"], [0, 0, 1], [1, 0, 0])
                 for f in range(3)]
        + [_finger([f"joint_{k}" for k in (12, 13, 14, 15)],
                   ["link_12", "link_13", "link_14", "link_15", "link_15_tip"],
                   [None, None, "thumb2", "thumb3", "tip"], [0, 0, 1], [1, 0, 0])],
        squeeze=[1, 1, 1, 2], tangential=[0, 0, 0, 1],
        clearance_meshes=[],
    ),
}
# Meshcat contact dots, one colour per finger.
CONTACT_COLORS = [Rgba(0.95, 0.15, 0.60, 1.0), Rgba(1.00, 0.55, 0.00, 1.0),
                  Rgba(1.00, 0.85, 0.00, 1.0), Rgba(0.10, 0.85, 0.85, 1.0)]
CONTACT_COLOR_NAMES = ["magenta", "orange", "yellow", "cyan"]


def use_hand(name):
    """Select the hand model. Sets the module-level description every other
    module reads: x = [NU hand joints (HAND_JOINTS order, finger by finger); q_o],
    u = hand joint increments."""
    global HAND, FINGERS, HAND_JOINTS, TIP_FRAMES, NU, LINK_INFO
    HAND = HANDS[name]
    FINGERS = range(len(HAND["fingers"]))
    HAND_JOINTS = [j for fg in HAND["fingers"] for j in fg["joints"]]
    TIP_FRAMES = [fg["links"][-1] for fg in HAND["fingers"]]
    NU = len(HAND_JOINTS)
    # link name -> (finger index, hand_spheres key or None)
    LINK_INFO = {link: (f, part) for f, fg in enumerate(HAND["fingers"])
                 for link, part in zip(fg["links"], fg["parts"])}


use_hand("dg3fm")


def joint_index(finger, k):
    """Index into x / u of joint k (0-3, base to tip) of finger `finger`."""
    return 4 * finger + k


def sphere_position(finger, spec):
    """(center in the link frame, radius) of a cfg["hand_spheres"] entry
    (dist, radius) or (dist, side offset, radius)."""
    dist, off, radius = spec if len(spec) == 3 else (spec[0], 0.0, spec[1])
    fg = HAND["fingers"][finger]
    return dist * fg["axis"] + off * fg["side"], radius


def is_free(cfg):
    """Free system: the object is a free body held only by the fingers (no
    support), q_o = [quaternion wxyz; position xyz]. Otherwise RotZ (Fig. 4a): q_o = yaw."""
    return cfg.get("system", "rotz") == "free"


def is_valve(cfg):
    """Object shape: a spoked valve (boxes) instead of the default sphere."""
    return cfg.get("object", "sphere") == "valve"


def n_q_object(cfg):
    return 7 if is_free(cfg) else 1


def n_v_object(cfg):
    return 6 if is_free(cfg) else 1


def object_center(cfg, x):
    """World position of the object center for state x."""
    if is_free(cfg):
        return np.asarray(x[NU + 4:NU + 7], float)
    return np.asarray(cfg["obj_center"], float)


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
            "- add_frame:\n    name: hand_mount\n    X_PF:\n      base_frame: world\n"
            f"      rotation: !Rpy {{deg: {[float(a) for a in HAND['weld_rpy_deg']]}}}\n"
            f"- add_weld:\n    parent: hand_mount\n    child: hand::{HAND['root']}\n")
    return files


def _write_hand_urdf(cfg, path, keep_visuals):
    ET.register_namespace("drake", DRAKE_NS)
    tree = ET.parse(HAND["urdf"])
    robot = tree.getroot()
    mesh_dir = HAND["mesh_dir"]
    sphere_specs = cfg["hand_spheres"]
    for dyn in robot.iter("dynamics"):
        dyn.attrib.pop("friction", None)  # joint friction: unsupported by Drake (warns)
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
        finger, part = LINK_INFO.get(name, (None, None))
        for i, spec in enumerate(sphere_specs.get(part, [])):
            center, radius = sphere_position(finger, spec)
            col = ET.SubElement(link, "collision", name=f"{name}_sphere{i}")
            ET.SubElement(col, "origin", xyz=" ".join(str(v) for v in center), rpy="0 0 0")
            ET.SubElement(ET.SubElement(col, "geometry"), "sphere", radius=str(radius))
            props = ET.SubElement(col, f"{{{DRAKE_NS}}}proximity_properties")
            for tag in ("mu_static", "mu_dynamic"):
                ET.SubElement(props, f"{{{DRAKE_NS}}}{tag}", value=str(cfg["mu"]))
    # Contacts in Eq. (2): hand-object and finger-finger (fingers cannot pass
    # through each other). Links of the same finger are filtered: their spheres
    # overlap by construction at the joints.
    for f in FINGERS:
        group = ET.SubElement(robot, f"{{{DRAKE_NS}}}collision_filter_group",
                              name=f"finger_{f}")
        for link in HAND["fingers"][f]["links"]:
            ET.SubElement(group, f"{{{DRAKE_NS}}}member", link=link)
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
    extra = ""  # further links of the object model (valve body)
    friction = f"""<drake:proximity_properties>
          <drake:mu_static>{cfg['mu']}</drake:mu_static>
          <drake:mu_dynamic>{cfg['mu']}</drake:mu_dynamic>
        </drake:proximity_properties>"""
    if is_valve(cfg):
        # Valve: valve_spokes boxes radiating from the hinge axis (box collision
        # geometry: sphere-box is the other pair qsim can differentiate), spoke 0
        # at valve_spoke_offset and drawn red, plus a visual-only hub.
        n, L, w, hgt = (cfg[k] for k in ("valve_spokes", "valve_spoke_length",
                                         "valve_spoke_width", "valve_height"))
        r = L  # size of the body-frame marker below
        shape = f"""
      <visual name="hub">
        <geometry><cylinder><radius>{0.8 * w}</radius><length>{1.1 * hgt}</length></cylinder></geometry>
        <material><diffuse>0.3 0.3 0.35 1</diffuse></material>
      </visual>"""
        for i in range(n):
            a = cfg["valve_spoke_offset"] + 2 * np.pi * i / n
            geom = (f"<pose>{L / 2 * np.cos(a)} {L / 2 * np.sin(a)} 0 0 0 {a}</pose>\n        "
                    f"<geometry><box><size>{L} {w} {hgt}</size></box></geometry>")
            shape += f"""
      <collision name="spoke_{i}">
        {geom}
        {friction}
      </collision>
      <visual name="spoke_{i}">
        {geom}
        <material><diffuse>{"0.9 0.1 0.1" if i == 0 else "0.2 0.4 0.9"} 1</diffuse></material>
      </visual>"""
        # Fixed valve body under the wheel (world-welded: two square boxes of
        # half-width valve_base_radius, one turned 45 deg, from z =
        # valve_base_bottom up to 2 mm below the wheel), so fingers cannot reach
        # in underneath it.
        if cfg.get("valve_base_radius"):
            rb, top = cfg["valve_base_radius"], cz - hgt / 2 - 0.002
            zb = cfg["valve_base_bottom"]
            base = ""
            for i, yaw in enumerate((0.0, np.pi / 4)):
                geom = (f"<pose>0 0 0 0 0 {yaw}</pose>\n        "
                        f"<geometry><box><size>{2 * rb} {2 * rb} {top - zb}</size></box></geometry>")
                base += f"""
      <collision name="base_{i}">
        {geom}
        {friction}
      </collision>
      <visual name="base_{i}">
        {geom}
        <material><diffuse>0.45 0.45 0.5 1</diffuse></material>
      </visual>"""
            extra = f"""
    <link name="valve_base">
      <pose>{cx} {cy} {(top + zb) / 2} 0 0 0</pose>{base}
    </link>
    <joint name="valve_base_weld" type="fixed">
      <parent>world</parent>
      <child>valve_base</child>
    </joint>
    <drake:collision_filter_group name="valve">
      <drake:member>object</drake:member>
      <drake:member>valve_base</drake:member>
      <drake:ignored_collision_filter_group>valve</drake:ignored_collision_filter_group>
    </drake:collision_filter_group>"""
    else:
        shape = f"""
      <collision name="object_sphere">
        <geometry><sphere><radius>{r}</radius></sphere></geometry>
        {friction}
      </collision>
      <visual name="object_visual">
        <geometry><sphere><radius>{r}</radius></sphere></geometry>
        <material><diffuse>0.2 0.4 0.9 0.6</diffuse></material>
      </visual>"""
    # Visual-only body frame at the object center: x red, y green, z blue. The
    # axes are 1.25 r long, so their ends stick out of the object and the
    # rotation is readable in Meshcat.
    length, t = 1.25 * r, 0.08 * r
    marker = ""
    for i, (axis, rgb) in enumerate(zip("xyz", ("0.9 0.1 0.1", "0.1 0.7 0.1", "0.1 0.2 0.9"))):
        size, pos = [t, t, t], [0.0, 0.0, 0.0]
        size[i], pos[i] = length, length / 2
        marker += f"""
      <visual name="axis_{axis}">
        <pose>{pos[0]} {pos[1]} {pos[2]} 0 0 0</pose>
        <geometry><box><size>{size[0]} {size[1]} {size[2]}</size></box></geometry>
        <material><diffuse>{rgb} 1</diffuse></material>
      </visual>"""
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
      </inertial>{shape}{marker}
    </link>{joint}{extra}
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
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(HAND["root"], hand),
                     RigidTransform(RollPitchYaw(np.radians(HAND["weld_rpy_deg"])), np.zeros(3)))
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
    """World position of finger `finger`'s contact sphere, hand_spheres["tip"][0]."""
    center, _ = sphere_position(finger, cfg["hand_spheres"]["tip"][0])
    frame = s["plant"].GetFrameByName(TIP_FRAMES[finger], s["hand"])
    return s["plant"].CalcPointsPositions(
        s["pctx"], frame, center, s["plant"].world_frame()).ravel()


def tip_contacts(s, cfg, x):
    """Per fingertip: signed distance tip sphere <-> object [m] and the closest
    point on the object surface (world frame). Sphere-sphere geometry, the same
    pair qsim uses (valve: closest pair between the tip link and any spoke)."""
    set_x(s, x)
    if is_valve(cfg):
        phi, points = np.full(len(FINGERS), np.inf), np.zeros((len(FINGERS), 3))
        qo = s["sg"].get_query_output_port().Eval(s["sg"].GetMyContextFromRoot(s["ctx"]))
        insp, plant = qo.inspector(), s["plant"]
        for sdp in qo.ComputeSignedDistancePairwiseClosestPoints(1.0):
            bodies = [plant.GetBodyFromFrameId(insp.GetFrameId(g)) for g in (sdp.id_A, sdp.id_B)]
            names = [b.name() for b in bodies]
            if "object" not in names:
                continue
            i_obj = names.index("object")
            tip = names[1 - i_obj]
            if tip in TIP_FRAMES and sdp.distance < phi[TIP_FRAMES.index(tip)]:
                j = TIP_FRAMES.index(tip)
                g, p_G = (sdp.id_A, sdp.p_ACa) if i_obj == 0 else (sdp.id_B, sdp.p_BCb)
                X_WG = plant.EvalBodyPoseInWorld(s["pctx"], bodies[i_obj]) @ insp.GetPoseInFrame(g)
                phi[j], points[j] = sdp.distance, X_WG.multiply(p_G)
        return phi, points
    c, r = object_center(cfg, x), cfg["obj_radius"]
    r_tip = cfg["hand_spheres"]["tip"][0][-1]
    phi, points = np.zeros(len(FINGERS)), np.zeros((len(FINGERS), 3))
    for j, f in enumerate(FINGERS):
        d = tip_sphere_center(s, f, cfg) - c
        dist = np.linalg.norm(d)
        phi[j] = dist - r - r_tip
        points[j] = c + r * d / dist
    return phi, points


def _add_contact_dots(s, cfg):
    for f in FINGERS:
        s["meshcat"].SetObject(f"contacts/tip{f + 1}", Sphere(cfg["contact_dot_radius"]),
                               CONTACT_COLORS[f])


def _set_contact_dots(s, cfg, x, time=None):
    """Move each finger's dot to its contact point; show it only while the tip
    is within cfg["contact_phi"]. With `time`, the change goes into the recording."""
    phi, points = tip_contacts(s, cfg, x)
    for j, f in enumerate(FINGERS):
        path = f"contacts/tip{f + 1}"
        s["meshcat"].SetTransform(path, RigidTransform(points[j]), time_in_recording=time)
        s["meshcat"].SetProperty(path, "visible", bool(phi[j] < cfg["contact_phi"]),
                                 time_in_recording=time)


def initial_grasp(s, cfg):
    """Each finger's tip sphere sits `grasp_gap` outside the sphere surface (or at
    distance grasp_radius from obj_center, if given) at
    azimuth finger_azimuth[f] and elevation grasp_elevation[f] (seen from the
    object center). Per finger, the joints in grasp_fixed[f] ({joint 0-3: value})
    are held and the others solved by IK from grasp_ik_guess[f], inside
    grasp_ik_bounds ({joint: (lo, hi)}, default: the joint limits). With more
    than 3 free joints, the tiny weight grasp_ik_reg picks the solution closest
    to the guess."""
    r_target = cfg["obj_radius"] + cfg["hand_spheres"]["tip"][0][-1] + cfg["grasp_gap"]
    if "grasp_radius" in cfg:  # tip-sphere center distance from obj_center, given directly
        r_target = cfg["grasp_radius"]
    c = np.asarray(cfg["obj_center"])
    q_lo, q_hi = joint_limits(s)
    reg = np.sqrt(cfg.get("grasp_ik_reg", 0.0))
    x = np.zeros(NU + n_q_object(cfg))
    if is_free(cfg):
        x[NU:] = [1.0, 0.0, 0.0, 0.0, *c]  # identity orientation at obj_center
    for f in FINGERS:
        azim, el = cfg["finger_azimuth"][f], cfg["grasp_elevation"][f]
        target = c + r_target * np.array(
            [np.cos(azim) * np.cos(el), np.sin(azim) * np.cos(el), np.sin(el)])
        q = np.array(cfg["grasp_ik_guess"][f], float)
        fixed = cfg["grasp_fixed"][f]
        q[list(fixed)] = list(fixed.values())
        free = [k for k in range(4) if k not in fixed]
        guess = q[free]
        lo, hi = q_lo[4 * f:4 * f + 4].copy(), q_hi[4 * f:4 * f + 4].copy()
        for k, (a, b) in cfg.get("grasp_ik_bounds", {}).items():
            lo[k], hi[k] = a, b

        def residual(q_free):
            q[free] = q_free
            x[4 * f:4 * f + 4] = q
            set_x(s, x)
            err = tip_sphere_center(s, f, cfg) - target
            return np.r_[err, reg * (q_free - guess)] if reg else err

        sol = least_squares(residual, guess, bounds=(lo[free], hi[free]),
                            xtol=1e-14, ftol=1e-14, gtol=1e-14)
        assert np.linalg.norm(sol.fun[:3]) < 1e-6, \
            f"grasp IK failed for finger {f + 1}: {sol.fun[:3]}"
        q[free] = sol.x
        x[4 * f:4 * f + 4] = q
    return x


def mesh_clearance(s, cfg, x):
    """Distance from the object sphere to the official (convex hulls of) hand
    collision meshes, vertex-based: a check that the simplified spheres do not
    hide a real penetration of the actual hand geometry (hands that ship
    collision meshes only)."""
    set_x(s, x)
    c, r = np.asarray(cfg["obj_center"]), cfg["obj_radius"]
    out = {}
    if is_valve(cfg):
        return out
    for link, stl in HAND["clearance_meshes"]:
        v = _load_stl(os.path.join(HAND["mesh_dir"], stl))
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
