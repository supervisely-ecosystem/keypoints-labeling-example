import supervisely as sly
from supervisely.geometry.graph import Node
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from dotenv import load_dotenv

# authentication
# # learn more here - https://developer.supervise.ly/getting-started/first-steps/basics-of-authentication
load_dotenv(os.path.expanduser('~/supervisely.env'))
api = sly.Api.from_env()
my_teams = api.team.get_list()
team = my_teams[0]
workspace = api.workspace.get_list(team.id)[0]

# create project and dataset
project = api.project.create(workspace.id, 'Surfer Pose Estimation', change_name_if_conflict=True)
dataset = api.dataset.create(project.id, 'Surfing', change_name_if_conflict=True)
print(f'Project {project.id} with dataset {dataset.id} are created')

# function for building geometry config
def build_config(n_keypoints: int, node_color: list, edge_color: list, skeleton: list):
  nodes = {}
  for i in range(n_keypoints):
    nodes[str(i)] = {'loc': [0, 0], 'color': node_color, 'disabled': False}
  edges = []
  for link in skeleton:
    edges.append({'dst': str(link[0]), 'src': str(link[1]), 'color': edge_color})
  return {'nodes': nodes, 'edges': edges}

# define geometry config parameters
n_keypoints = 17 # human body has 17 keypoints
color = [255, 0, 0] # red
skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], # links between nodes (e.g. node 15 is linked to node 13)
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
            [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
            [3, 5], [4, 6]]
config = build_config(n_keypoints, node_color=color, edge_color=color, skeleton=skeleton)

# create annotation class
surfer = sly.ObjClass('surfer',
                      geometry_type=sly.GraphNodes,
                      geometry_config=config)
project_meta = sly.ProjectMeta(obj_classes=[surfer])
api.project.update_meta(project.id, project_meta.to_json())

# upload image to the dataset
image_info = api.image.upload_path(dataset.id,
                                   name='Surfer.jpg',
                                   path='images/Surfer.jpg')

# create annotation and upload to image
# build keypoints graph
vertex_0 = Node(sly.PointLocation(364, 775), disabled=False)
vertex_1 = Node(sly.PointLocation(353, 773), disabled=False)
vertex_2 = Node(sly.PointLocation(352, 766), disabled=False)
vertex_3 = Node(sly.PointLocation(358, 736), disabled=False)
vertex_4 = Node(sly.PointLocation(357, 725), disabled=False)
vertex_5 = Node(sly.PointLocation(447, 747), disabled=False)
vertex_6 = Node(sly.PointLocation(403, 649), disabled=False)
vertex_7 = Node(sly.PointLocation(531, 767), disabled=False)
vertex_8 = Node(sly.PointLocation(448, 545), disabled=False)
vertex_9 = Node(sly.PointLocation(585, 782), disabled=False)
vertex_10 = Node(sly.PointLocation(538, 507), disabled=False)
vertex_11 = Node(sly.PointLocation(567, 624), disabled=False)
vertex_12 = Node(sly.PointLocation(545, 576), disabled=False)
vertex_13 = Node(sly.PointLocation(700, 668), disabled=False)
vertex_14 = Node(sly.PointLocation(646, 704), disabled=False)
vertex_15 = Node(sly.PointLocation(769, 547), disabled=False)
vertex_16 = Node(sly.PointLocation(802, 665), disabled=False)
nodes = {'0': vertex_0, '1': vertex_1, '2': vertex_2, '3': vertex_3, '4': vertex_4,
         '5': vertex_5, '6': vertex_6, '7': vertex_7, '8': vertex_8, '9': vertex_9,
         '10': vertex_10, '11': vertex_11, '12': vertex_12, '13': vertex_13, '14': vertex_14,
         '15': vertex_15, '16': vertex_16}


# label the image
label = sly.Label(sly.GraphNodes(nodes), surfer)
ann = sly.Annotation(img_size=[1280, 1920], labels=[label])
api.annotation.upload_ann(image_info.id, ann)

# download image and annotation
image = api.image.download_np(image_info.id)
print('image shape (height, width, channels)', image.shape)

ann_json = api.annotation.download_json(image_info.id) 
print('annotaiton:\n', json.dumps(ann_json, indent=4))

# draw annotation
ann = sly.Annotation.from_json(ann_json, project_meta)
output_path = 'images/Labeled.jpg'
ann.draw_pretty(image, output_path=output_path, thickness=3)

# function for image visualization
def visualize_image(image_filepath):
  plt.figure(figsize=(12, 8))
  image = mpimg.imread(image_filepath)
  imageplot = plt.imshow(image)
  plt.axis('off')
  plt.show()

# visualize result
visualize_image(output_path)