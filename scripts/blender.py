# NOTE that you need to install pytorch=3.10 in order to run the script
# This does not work with the main environment
# Please additionally install bpy and trimesh

import math
import os
from pathlib import Path

import bpy
import trimesh

temp_folder = Path("/home/borth/sketch2shape/temp")
blender_path = temp_folder / "video.blend"
source_folder = temp_folder / "marlene"
mesh_folder = source_folder / "mesh"
output_folder = source_folder / "output"
stl_folder = source_folder / "stls"

# Iterate over all objs in the folder
for filename in sorted(os.listdir(mesh_folder)):
    if filename.endswith(".obj"):
        mesh = trimesh.load_mesh(mesh_folder / filename)

        # Convert and save to .stl file
        stl_data = mesh.export(file_type="stl")
        name = filename.split(".")[0]
        stl_filename = stl_folder / f"{name}.stl"
        stl_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(str(stl_filename), "wb") as f:
            f.write(stl_data)

        # Load the Blender file
        bpy.ops.wm.open_mainfile(filepath=str(blender_path.resolve()))

        # Select the object to be replaced
        object_to_replace = bpy.data.objects["chair.001"]
        bpy.context.collection.objects.unlink(object_to_replace)

        # Load the replacement object
        bpy.ops.import_mesh.stl(filepath=str(stl_filename))

        # Set the position and rotation of the replacement object
        replacement_object = bpy.context.selected_objects[0]
        replacement_object.location = object_to_replace.location
        replacement_object.rotation_euler = object_to_replace.rotation_euler
        replacement_object.active_material = object_to_replace.active_material

        # Render the image
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images["Render Result"]

        # Specify the path where you want to save the image
        output_path = output_folder / f"rendered/{filename.split('.')[0]}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save_render(str(output_path.resolve()))

    # Number of frames to capture
    num_frames = 72
    # angle_increment = 360 / num_frames
    for i in range(num_frames):
        replacement_object.rotation_euler.z = math.radians(
            270 - (i * (360 / num_frames))
        )
        bpy.ops.render.render(write_still=True)
        output_path = (
            output_folder / f"video_frames/{filename.split('.')[0]}/{i:05}.png"
        )
        img.save_render(str(output_path))

    framerate = 30
    video_path = output_folder / f"video/{filename.split('.')[0]}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    image_folder_glob = (
        output_folder / f"video_frames/{filename.split('.')[0]}/*.png"
    ).as_posix()
    args: list[str] = [
        f"ffmpeg -framerate {framerate}",
        f'-pattern_type glob -i "{image_folder_glob}"',
        f"-pix_fmt yuv420p {str(video_path)}",
        "-y",
    ]
    os.system(" ".join(args))
