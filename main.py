import supervisely as sly
from supervisely.geometry.graph import Node, KeypointsTemplate
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from dotenv import load_dotenv

# authentication
# # learn more here - https://developer.supervise.ly/getting-started/first-steps/basics-of-authentication
load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api.from_env()
my_teams = api.team.get_list()
team = my_teams[0]
workspace = api.workspace.get_list(team.id)[0]

# create project and dataset
project = api.project.create(
    workspace.id, "Surfer Pose Estimation", change_name_if_conflict=True
)
dataset = api.dataset.create(project.id, "Surfing", change_name_if_conflict=True)
print(f"Project {project.id} with dataset {dataset.id} are created")

# build geometry config
# initialize template
template = KeypointsTemplate()
# add nodes
template.add_point(label="nose", row=515, col=248, color=[255, 0, 0])
template.add_point(label="left_eye", row=536, col=230, color=[255, 0, 0])
template.add_point(label="right_eye", row=496, col=228, color=[255, 0, 0])
template.add_point(label="left_ear", row=561, col=248, color=[255, 0, 0])
template.add_point(label="right_ear", row=464, col=242, color=[255, 0, 0])
template.add_point(label="left_shoulder", row=615, col=367, color=[255, 0, 0])
template.add_point(label="right_shoulder", row=400, col=383, color=[255, 0, 0])
template.add_point(label="left_elbow", row=653, col=559, color=[255, 0, 0])
template.add_point(label="right_elbow", row=364, col=554, color=[255, 0, 0])
template.add_point(label="left_wrist", row=655, col=712, color=[255, 0, 0])
template.add_point(label="right_wrist", row=364, col=711, color=[255, 0, 0])
template.add_point(label="left_hip", row=565, col=703, color=[255, 0, 0])
template.add_point(label="right_hip", row=437, col=705, color=[255, 0, 0])
template.add_point(label="left_knee", row=568, col=922, color=[255, 0, 0])
template.add_point(label="right_knee", row=434, col=919, color=[255, 0, 0])
template.add_point(label="left_ankle", row=571, col=1109, color=[255, 0, 0])
template.add_point(label="right_ankle", row=442, col=1100, color=[255, 0, 0])
# add edges
template.add_edge(src="left_ankle", dst="left_knee", color=[255, 0, 0])
template.add_edge(src="left_knee", dst="left_hip", color=[255, 0, 0])
template.add_edge(src="right_ankle", dst="right_knee", color=[255, 0, 0])
template.add_edge(src="right_knee", dst="right_hip", color=[255, 0, 0])
template.add_edge(src="left_hip", dst="right_hip", color=[255, 0, 0])
template.add_edge(src="left_shoulder", dst="left_hip", color=[255, 0, 0])
template.add_edge(src="right_shoulder", dst="right_hip", color=[255, 0, 0])
template.add_edge(src="left_shoulder", dst="right_shoulder", color=[255, 0, 0])
template.add_edge(src="left_shoulder", dst="left_elbow", color=[255, 0, 0])
template.add_edge(src="right_shoulder", dst="right_elbow", color=[255, 0, 0])
template.add_edge(src="left_elbow", dst="left_wrist", color=[255, 0, 0])
template.add_edge(src="right_elbow", dst="right_wrist", color=[255, 0, 0])
template.add_edge(src="left_eye", dst="right_eye", color=[255, 0, 0])
template.add_edge(src="nose", dst="left_eye", color=[255, 0, 0])
template.add_edge(src="nose", dst="right_eye", color=[255, 0, 0])
template.add_edge(src="left_eye", dst="left_ear", color=[255, 0, 0])
template.add_edge(src="right_eye", dst="right_ear", color=[255, 0, 0])
template.add_edge(src="left_ear", dst="left_shoulder", color=[255, 0, 0])
template.add_edge(src="right_ear", dst="right_shoulder", color=[255, 0, 0])

# create annotation class
surfer = sly.ObjClass("surfer", geometry_type=sly.GraphNodes, geometry_config=template)
project_meta = sly.ProjectMeta(obj_classes=[surfer])
api.project.update_meta(project.id, project_meta.to_json())

# upload image to the dataset
image_info = api.image.upload_path(
    dataset.id, name="Surfer.jpg", path="images/Surfer.jpg"
)

# create annotation and upload to image
# build keypoints graph
v1 = sly.Node(label="nose", row=364, col=775)
v2 = sly.Node(label="left_eye", row=353, col=773)
v3 = sly.Node(label="right_eye", row=352, col=766)
v4 = sly.Node(label="left_ear", row=358, col=736)
v5 = sly.Node(label="right_ear", row=357, col=752)
v6 = sly.Node(label="left_shoulder", row=447, col=747)
v7 = sly.Node(label="right_shoulder", row=403, col=649)
v8 = sly.Node(label="left_elbow", row=531, col=767)
v9 = sly.Node(label="right_elbow", row=448, col=545)
v10 = sly.Node(label="left_wrist", row=585, col=782)
v11 = sly.Node(label="right_wrist", row=538, col=507)
v12 = sly.Node(label="left_hip", row=567, col=624)
v13 = sly.Node(label="right_hip", row=545, col=576)
v14 = sly.Node(label="left_knee", row=700, col=668)
v15 = sly.Node(label="right_knee", row=646, col=704)
v16 = sly.Node(label="left_ankle", row=769, col=547)
v17 = sly.Node(label="right_ankle", row=802, col=665)

nodes = [
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    v8,
    v9,
    v10,
    v11,
    v12,
    v13,
    v14,
    v15,
    v16,
    v17,
]


# label the image
label = sly.Label(sly.GraphNodes(nodes), surfer)
ann = sly.Annotation(img_size=[1280, 1920], labels=[label])
api.annotation.upload_ann(image_info.id, ann)

# download image and annotation
image = api.image.download_np(image_info.id)
print("image shape (height, width, channels)", image.shape)

ann_json = api.annotation.download_json(image_info.id)
print("annotaiton:\n", json.dumps(ann_json, indent=4))

# draw annotation
ann = sly.Annotation.from_json(ann_json, project_meta)
output_path = "images/Labeled.jpg"
ann.draw_pretty(image, output_path=output_path, thickness=3)

# function for image visualization
def visualize_image(image_filepath):
    plt.figure(figsize=(12, 8))
    image = mpimg.imread(image_filepath)
    imageplot = plt.imshow(image)
    plt.axis("off")
    plt.show()


# visualize result
visualize_image(output_path)
