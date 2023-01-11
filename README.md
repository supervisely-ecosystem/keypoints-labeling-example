# Supervisely Data Labelling Example: Keypoints

In this tutorial we will show you how to use sly.GraphNodes class to create data annotation for pose estimation / keypoints detection task. The tutorial illustrates basic upload-download scenario:

* create project and dataset on server
* upload image
* programmatically create annotation (two bounding boxes and tag) and upload it to image
* download image and annotation

You can try this example for yourself: VSCode project config, original image, and python script for this tutorial are ready on [GitHub](https://github.com/supervisely-ecosystem/keypoints-labelling-example).

## Installation & Authentication

Run the following command:
```
pip install supervisely
```

Import necessary libraries:

```python
import supervisely as sly
from supervisely.geometry.graph import Node
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from dotenv import load_dotenv
```

Authenticate:
```python
load_dotenv(os.path.expanduser('/content/supervisely.env'))
api = sly.Api.from_env()
my_teams = api.team.get_list()
team = my_teams[0]
workspace = api.workspace.get_list(team.id)[0]
```

## Input Data
![Surfer](https://user-images.githubusercontent.com/91027877/211779545-83935382-b8a2-49cb-9156-3ce07f902399.jpg)

## Create Project On Server & Build Geometry Config

Create project and dataset:
```python
project = api.project.create(workspace.id, 'Surfer Pose Estimation', change_name_if_conflict=True)
dataset = api.dataset.create(project.id, 'Surfing', change_name_if_conflict=True)
print(f'Project {project.id} with dataset {dataset.id} are created')
```

Function for building geometry config:
```python
def build_config(n_keypoints: int, node_color: list, edge_color: list, skeleton: list):
  nodes = {}
  for i in range(n_keypoints):
    nodes[str(i)] = {'loc': [0, 0], 'color': node_color, 'disabled': False}
  edges = []
  for link in skeleton:
    edges.append({'dst': str(link[0]), 'src': str(link[1]), 'color': edge_color})
  return {'nodes': nodes, 'edges': edges}
```

Define geometry config parameters:
```python
n_keypoints = 17 # human body has 17 keypoints
color = [255, 0, 0] # red
skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], # links between nodes (e.g. node 15 is linked to node 13)
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
            [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
            [3, 5], [4, 6]]
config = build_config(n_keypoints, node_color=color, edge_color=color, skeleton=skeleton)
```

Create annotation class:
```python
surfer = sly.ObjClass('surfer',
                      geometry_type=sly.GraphNodes,
                      geometry_config=config)
project_meta = sly.ProjectMeta(obj_classes=[surfer])
api.project.update_meta(project.id, project_meta.to_json())
```

## Upload Image
```python
image_info = api.image.upload_path(dataset.id,
                                   name='Surfer.jpg',
                                   path='images/Surfer.jpg')
```
 
 ## Create Annotation & Upload To Image
 
 Build keypoints graph:
 ```python
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
```
Label the image:
```python
label = sly.Label(sly.GraphNodes(nodes), surfer)
ann = sly.Annotation(img_size=[1280, 1920], labels=[label])
api.annotation.upload_ann(image_info.id, ann)
```

## Download Data

```python
image = api.image.download_np(image_info.id)
print('image shape (height, width, channels)', image.shape)

ann_json = api.annotation.download_json(image_info.id) 
print('annotaiton:\n', json.dumps(ann_json, indent=4))
```

## Result

![Labeled](https://user-images.githubusercontent.com/91027877/211782477-fa09bfbb-82b3-47ba-b86e-0187726e294f.jpg)
