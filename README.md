# Supervisely Data Labelling Example: Keypoints

In this tutorial we will show you how to use sly.GraphNodes class to create data annotation for pose estimation / keypoints detection task. The tutorial illustrates basic upload-download scenario:

* create project and dataset on server
* upload image
* programmatically create annotation (two bounding boxes and tag) and upload it to image
* download image and annotation

## Installation & Authentication

Run the following command in terminal:
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

Authenticate (learn more [here](https://developer.supervise.ly/getting-started/first-steps/basics-of-authentication)):
```python
load_dotenv(os.path.expanduser('~/supervisely.env'))
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

Build geometry config :
```python
color = [255, 0, 0] # red
config = sly.GraphNodes.build_config(node_color=color, edge_color=color)
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

## Visualize Result

Draw annotation:
```python
ann = sly.Annotation.from_json(ann_json, project_meta)
output_path = 'images/Labeled.jpg'
ann.draw_pretty(image, output_path=output_path, thickness=3)
```

Function for image visualization:
```python
def visualize_image(image_filepath):
  plt.figure(figsize=(12, 8))
  image = mpimg.imread(image_filepath)
  imageplot = plt.imshow(image)
  plt.axis('off')
  plt.show()
```

Visualize result:
```python
visualize_image(output_path)
```
If you are facing "Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure" UserWarning, run the following comand in terminal:
```bash
sudo apt-get install python3-tk
```


![Labeled](https://user-images.githubusercontent.com/91027877/211782477-fa09bfbb-82b3-47ba-b86e-0187726e294f.jpg)

## Working With Keypoints In Annotation Tool

Example above shows how to programmaticaly create keypoints annotation, but you can also use Supervisely annotation tool.

GIFs below demonstrate keypoints graph template creation algorithm in annotation tool.

1. Open your dataset, choose image, click on "Add keypoints" button in annotation tool, give your graph template a name, choose color and upload image for creating graph template:
![keypoints_0](https://user-images.githubusercontent.com/91027877/212082648-72ca9cf2-0033-4aac-a1b7-211fa13030e1.gif)

2. Start creating graph tempate by setting keypoints (nodes) on the image (you can change the size of nodes with the help of mouse scroll wheel):
![keypoints_1](https://user-images.githubusercontent.com/91027877/212090976-353736f2-5d47-45ea-bc5d-a5c3756e21b0.gif)

3. Link nodes with each other using edges:
![keypoints_2](https://user-images.githubusercontent.com/91027877/212097299-f607c339-7f10-4c32-a85f-f5ff48877743.gif)

4. Save result and go back to annotation tool:
![keypoints_3](https://user-images.githubusercontent.com/91027877/212101921-a9825133-d2ee-4945-88ad-48630700d507.gif)

5. Now you can use your graph template to label the image. Using templates can significantly speed up data labelling process: it is much easier to create one template and then tune only its width and height instead of building keypoints graph for each object on the image:
![keypoints_4](https://user-images.githubusercontent.com/91027877/212112189-b35c15f9-7d63-4386-8058-528381aacf79.gif)

6. Result

![image](https://user-images.githubusercontent.com/91027877/212114675-8dda2cdf-e8ac-437d-aab5-9aadcc8b1ce5.png)

