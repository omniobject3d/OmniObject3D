import sys, os
import json
import bpy
import mathutils
import numpy as np


DEBUG = False

VIEWS = 100
RESOLUTION = 800
#RESULTS_PATH = 'results'
DEPTH_SCALE = 0.1 # 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
DEPTH_FORMAT = 'OPEN_EXR'
RANDOM_VIEWS = True
UPPER_VIEWS = True
CIRCLE_FIXED_START = (.3,0,0)
engine = 'BLENDER_EEVEE'

def get_scale(obj):
    maxWCoord = [0,0,0]
    WCoord = [[], [], []]
    for vert in obj.data.vertices:
        wmtx = obj.matrix_world
        worldCoord = vert.co @ wmtx
        WCoord[0] += [np.abs(worldCoord[0])]
        WCoord[1] += [np.abs(worldCoord[1])]
        WCoord[2] += [np.abs(worldCoord[2])]
    if len(WCoord[0]) == 0:
        return None
    maxWCoord[0] = np.max(WCoord[0])
    maxWCoord[1] = np.max(WCoord[1])
    maxWCoord[2] = np.max(WCoord[2])
    max_coord = np.max(maxWCoord)
    scale = 0.99/max_coord
    return scale

def getGeometryCenter(obj):
    sumWCoord = [0,0,0]
    numbVert = 0
    if obj.type == 'MESH':
        for vert in obj.data.vertices:
            wmtx = obj.matrix_world
            worldCoord = vert.co @ wmtx
            sumWCoord[0] += worldCoord[0]
            sumWCoord[1] += worldCoord[1]
            sumWCoord[2] += worldCoord[2]
            numbVert += 1
        sumWCoord[0] = sumWCoord[0]/numbVert
        sumWCoord[1] = sumWCoord[1]/numbVert
        sumWCoord[2] = sumWCoord[2]/numbVert
    return sumWCoord

def getGeometryBound(obj):

    return sumWCoord

def setOrigin(obj, r):
    oldLoc = obj.location
    newLoc = getGeometryCenter(obj)
    obj.location.x += (newLoc[0] - oldLoc[0]) * r
    obj.location.y += (newLoc[1] - oldLoc[1]) * r
    obj.location.z += (newLoc[2] - oldLoc[2]) * r

def scaleObject(obj, r, adaptive=False):
    obj.scale.x *= r
    obj.scale.y *= r
    obj.scale.z *= r

##Cleans all decimate modifiers
def cleanAllDecimateModifiers(obj):
    for m in obj.modifiers:
        if(m.type=="DECIMATE"):
            #           print("Removing modifier ")
            obj.modifiers.remove(modifier=m)

def render_once(RESULTS_PATH, scale):
    fp = bpy.path.abspath(f"//{RESULTS_PATH}")

    def listify_matrix(matrix):
        matrix_list = []
        for row in matrix:
            matrix_list.append(list(row))
        return matrix_list

    if not os.path.exists(fp):
        os.makedirs(fp)

    # Data to store in JSON file
    out_data = {
        'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
    }

    # Render Optimizations
    bpy.context.scene.render.use_persistent_data = True

    context = bpy.context
    scene = bpy.context.scene
    render = bpy.context.scene.render

    render.engine = engine
    #    # Set the device_type
    #    context.preferences.addons[
    #        "cycles"
    #    ].preferences.compute_device_type = "CUDA" # or "OPENCL"
    #    # Set the device and feature set
    #    scene.cycles.device = "GPU"

    render.image_settings.color_mode = 'RGBA' # ('RGB', 'RGBA', ...)
    render.image_settings.color_depth = str(COLOR_DEPTH)
    render.image_settings.file_format = str(FORMAT)
    render.resolution_x = RESOLUTION
    render.resolution_y = RESOLUTION
    render.resolution_percentage = 100
    render.film_transparent = True

    # Set up rendering of depth map.
    scene.use_nodes = True
    scene.view_layers["View Layer"].use_pass_normal = True
    scene.view_layers["View Layer"].use_pass_diffuse_color = True
    scene.view_layers["View Layer"].use_pass_object_index = True

    tree = bpy.context.scene.node_tree
    nodes = tree.nodes
    links = tree.links
    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    #bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
    # bpy.context.view_layer.use_pass_z = True

    if not DEBUG:
        # Create input render layer node.
        render_layers = tree.nodes.new('CompositorNodeRLayers')

        depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.label = 'Depth Output'
        depth_file_output.base_path = ''
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = DEPTH_FORMAT
        # depth_file_output.format.color_depth = str(COLOR_DEPTH)
        if DEPTH_FORMAT == 'OPEN_EXR':
            links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
        else:
            depth_file_output.format.color_mode = "BW"
            # Remap as other types can not represent the full range of depth.
            map = nodes.new(type="CompositorNodeMapValue")
            # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
            map.offset = [-0.7]
            map.size = [DEPTH_SCALE]
            map.use_min = True
            map.min = [0]
            links.new(render_layers.outputs['Depth'], map.inputs[0])
            links.new(map.outputs[0], depth_file_output.inputs[0])

        normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = 'Normal Output'
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

    # Background
    bpy.context.scene.render.dither_intensity = 0.0
    bpy.context.scene.render.film_transparent = True

    # Create collection for objects not to render with background


    objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
    bpy.ops.object.delete({"selected_objects": objs})

    def parent_obj_to_camera(b_camera):
        origin = (0, 0, 0)
        b_empty = bpy.data.objects.new("Empty", None)
        b_empty.location = origin
        b_camera.parent = b_empty  # setup parenting

        scn = bpy.context.scene
        scn.collection.objects.link(b_empty)
        bpy.context.view_layer.objects.active = b_empty
        # scn.objects.active = b_empty
        return b_empty


    scene = bpy.context.scene
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100

    cam = scene.objects['Camera']
    cam.location = (0, 4.0, 0.5)
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty

    scene.render.image_settings.file_format = 'PNG'  # set output format to .png

    from math import radians

    stepsize = 360.0 / VIEWS
    rotation_mode = 'XYZ'

    if not DEBUG:
        for output_node in [depth_file_output, normal_file_output]:
            output_node.base_path = ''

    out_data['frames'] = []

    if not RANDOM_VIEWS:
        b_empty.rotation_euler = CIRCLE_FIXED_START

    for i in range(0, VIEWS):
        if DEBUG:
            i = np.random.randint(0,VIEWS)
            b_empty.rotation_euler[2] += radians(stepsize*i)
        if RANDOM_VIEWS:
            scene.render.filepath = fp + '/r_' + str(i)
            if UPPER_VIEWS:
                rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
                rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
                b_empty.rotation_euler = rot
            else:
                b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
        else:
            print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
            scene.render.filepath = fp + '/r_' + str(i)

        depth_file_output.file_slots[0].path = scene.render.filepath + "_depth_"
        normal_file_output.file_slots[0].path = scene.render.filepath + "_normal_"

        if DEBUG:
            break
        else:
            bpy.ops.render.render(write_still=True)  # render still

        frame_data = {
            'file_path': scene.render.filepath,
            'rotation': radians(stepsize),
            'transform_matrix': listify_matrix(cam.matrix_world),
            'scale': scale
        }
        out_data['frames'].append(frame_data)

        if RANDOM_VIEWS:
            if UPPER_VIEWS:
                rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
                rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
                b_empty.rotation_euler = rot
            else:
                b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
        else:
            b_empty.rotation_euler[2] += radians(stepsize)

    if not DEBUG:
        with open(fp + '/' + 'transforms.json', 'w') as out_file:
            json.dump(out_data, out_file, indent=4)

if "Cube" in bpy.data.objects:
    bpy.data.objects["Cube"].hide_render = True
    bpy.data.objects["Cube"].hide_viewport = True

# Make light just directional, disable shadows.
light = bpy.data.lights['Light']
light.type = 'SUN'
light.use_shadow = False
# Possibly disable specular shading:
light.specular_factor = 1.0
light.energy = 10.0

# Add another light source so stuff facing away from light is not completely dark
bpy.ops.object.light_add(type='SUN')
light2 = bpy.data.lights['Sun']
light2.use_shadow = False
light2.specular_factor = 1.0
light2.energy = 1
bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Light'].rotation_euler
bpy.data.objects['Sun'].rotation_euler[0] += 180

root="the_root_of_raw_scans"
cats = sorted(os.listdir(root))
for cat in cats:
    if os.path.isfile(os.path.join(root, cat)):
        continue
    files = sorted(os.listdir(os.path.join(root, cat)))
    for file in files:
        scan_files = os.listdir(os.path.join(root, cat, file, 'Scan'))
        for scan_file in scan_files:
            if '.obj' not in scan_file:
                continue
            filepath = os.path.join(root, cat, file, 'Scan', scan_file)
            render_path = os.path.join(root, cat, file, 'render')
#            if os.path.exists(render_path):
#                continue
            bpy.ops.import_scene.obj(filepath=filepath)
            scene = bpy.context.scene
            mesh_obs = [o for o in scene.objects if o.type == 'MESH']
            
            # remove "Cube" if it exists.
            if "Cube" in bpy.data.objects:
                obj = mesh_obs[1]
            else:
                obj = mesh_obs[0]

            scale = get_scale(obj)
            if scale is None:
                bpy.ops.object.delete()
                continue
            scaleObject(obj, scale)
            # setOrigin(obj, scale)

            obj.select_set(True) # Blender 2.8x
            render_once(render_path, scale)
            bpy.ops.object.delete()
