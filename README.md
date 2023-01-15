# Supervisely Data Labelling Example: Keypoints

In this tutorial we will show you how to use sly.GraphNodes class to create data annotation for pose estimation / keypoints detection task. The tutorial illustrates basic upload-download scenario:

* create project and dataset on server
* upload image
* programmatically create annotation (two bounding boxes and tag) and upload it to image
* download image and annotation

## Installation & Importing Necessary Libraries

Run the following command in terminal:
```
pip install supervisely
```

Import necessary libraries:

```python
import supervisely as sly
from supervisely.geometry.graph import Node, KeypointsTemplate
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
from dotenv import load_dotenv
```

Before we will start creating our project, let's learn how to create keypoints template - we are going to use it in our project.

## Working With Keypoints Template

We will need an image to create and visualize our keypoints template.

In tutorial we are going to visualize images often, so let's write a simple function for image visualozation:
```python
def visualize_image(image_filepath):
  plt.figure(figsize=(12, 8))
  image = mpimg.imread(image_filepath)
  imageplot = plt.imshow(image)
  plt.axis('off')
  plt.show()
```

Visualize image for building keypoints template:
```python
visualize_image("images/girl.jpg")
```
![girl](https://user-images.githubusercontent.com/91027877/212552404-f6f6a93c-ff15-43ba-ab24-32a71957bb9f.jpg)

Create keypoints template:
```python
# initialize template
template = KeypointsTemplate()
# add nodes
template.add_point(label="nose", row=635, col=427)
template.add_point(label="left_eye", row=597, col=404)
template.add_point(label="right_eye", row=685, col=401)
template.add_point(label="left_ear", row=575, col=431)
template.add_point(label="right_ear", row=723, col=425)
template.add_point(label="left_shoulder", row=502, col=614)
template.add_point(label="right_shoulder", row=794, col=621)
template.add_point(label="left_elbow", row=456, col=867)
template.add_point(label="right_elbow", row=837, col=874)
template.add_point(label="left_wrist", row=446, col=1066)
template.add_point(label="right_wrist", row=845, col=1073)
template.add_point(label="left_hip", row=557, col=1035)
template.add_point(label="right_hip", row=743, col=1043)
template.add_point(label="left_knee", row=541, col=1406)
template.add_point(label="right_knee", row=751, col=1421)
template.add_point(label="left_ankle", row=501, col=1760)
template.add_point(label="right_ankle", row=774, col=1765)
# add edges
template.add_edge(src="left_ankle", dst="left_knee")
template.add_edge(src="left_knee", dst="left_hip")
template.add_edge(src="right_ankle", dst="right_knee")
template.add_edge(src="right_knee", dst="right_hip")
template.add_edge(src="left_hip", dst="right_hip")
template.add_edge(src="left_shoulder", dst="left_hip")
template.add_edge(src="right_shoulder", dst="right_hip")
template.add_edge(src="left_shoulder", dst="right_shoulder")
template.add_edge(src="left_shoulder", dst="left_elbow")
template.add_edge(src="right_shoulder", dst="right_elbow")
template.add_edge(src="left_elbow", dst="left_wrist")
template.add_edge(src="right_elbow", dst="right_wrist")
template.add_edge(src="left_eye", dst="right_eye")
template.add_edge(src="nose", dst="left_eye")
template.add_edge(src="nose", dst="right_eye")
template.add_edge(src="left_eye", dst="left_ear")
template.add_edge(src="right_eye", dst="right_ear")
template.add_edge(src="left_ear", dst="left_shoulder")
template.add_edge(src="right_ear", dst="right_shoulder")
```

Visualize your keypoints template:
```python
template_img = sly.image.read("images/girl.jpg")
template.draw(image=template_img, thickness=7)
sly.image.write("images/template.jpg", template_img)
visualize_image("images/template.jpg")
```
![template](https://user-images.githubusercontent.com/91027877/212552424-87a0c197-63ce-46d1-95b7-aefd640076a8.jpg)

You can also transfer your template to json:
```python
template_json = template.to_json()
```

Now, when we have successfully created keypoints template, we can start creating keypoints annotation for our project.

## Programmatically Create Keypoints Annotation

Authenticate (learn more [here](https://developer.supervise.ly/getting-started/first-steps/basics-of-authentication)):
```python
load_dotenv(os.path.expanduser('~/supervisely.env'))
api = sly.Api.from_env()
my_teams = api.team.get_list()
team = my_teams[0]
workspace = api.workspace.get_list(team.id)[0]
```

Visualize input image:
```python
visualize_image("images/person_with_dog.jpg")
```
![person_with_dog](https://user-images.githubusercontent.com/91027877/212552599-294c41aa-72bb-4243-8a41-5fbfc73f9a0e.jpg)

Create project and dataset:
```python
project = api.project.create(workspace.id, "Human Pose Estimation", change_name_if_conflict=True)
dataset = api.dataset.create(project.id, "Person with dog", change_name_if_conflict=True)
print(f"Project {project.id} with dataset {dataset.id} are created")
```

Now let's create annotation class using our keypoints template as a geometry config (unlike other supervisely geometry classes, sly.GraphNodes requires geometry config to be passed - it is necessary for object class initialization):
```python
person = sly.ObjClass("person", geometry_type=sly.GraphNodes, geometry_config=template)
project_meta = sly.ProjectMeta(obj_classes=[person])
api.project.update_meta(project.id, project_meta.to_json())
```

Upload image:
```python
image_info = api.image.upload_path(
    dataset.id, name="person_with_dog.jpg", path="images/person_with_dog.jpg"
)
```

Build keypoints graph:
```python
nodes = [
    sly.Node(label="nose", row=146, col=702),
    sly.Node(label="left_eye", row=130, col=644),
    sly.Node(label="right_eye", row=135, col=701),
    sly.Node(label="left_ear", row=137, col=642),
    sly.Node(label="right_ear", row=142, col=705),
    sly.Node(label="left_shoulder", row=221, col=595),
    sly.Node(label="right_shoulder", row=226, col=738),
    sly.Node(label="left_elbow", row=335, col=564),
    sly.Node(label="right_elbow", row=342, col=765),
    sly.Node(label="left_wrist", row=429, col=555),
    sly.Node(label="right_wrist", row=438, col=784),
    sly.Node(label="left_hip", row=448, col=620),
    sly.Node(label="right_hip", row=451, col=713),
    sly.Node(label="left_knee", row=598, col=591),
    sly.Node(label="right_knee", row=602, col=715),
    sly.Node(label="left_ankle", row=761, col=573),
    sly.Node(label="right_ankle", row=766, col=709),
]
```

Label the image:
```python
input_image = sly.image.read("images/person_with_dog.jpg")
img_height, img_width = input_image.shape[:2]
label = sly.Label(sly.GraphNodes(nodes), person)
ann = sly.Annotation(img_size=[img_height, img_width], labels=[label])
api.annotation.upload_ann(image_info.id, ann)
```

Download data:
```python
image = api.image.download_np(image_info.id)
ann_json = api.annotation.download_json(image_info.id)
```

Draw annotation:
```python
ann = sly.Annotation.from_json(ann_json, project_meta)
output_path = "images/person_with_dog_labelled.jpg"
ann.draw_pretty(image, output_path=output_path, thickness=7)
```

Visualize result:
```python
visualize_image(output_path)
```
![person_with_dog_labelled](https://user-images.githubusercontent.com/91027877/212553004-8062ad93-c1da-46e9-ab53-956f4e947f87.jpg)







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

5. Now you can use your graph template to label the image. Using templates can significantly speed up data labelling process: it is much easier to create one template and then tune only its width and height instead of building keypoints graph for each object on the image from scratch:
![keypoints_4](https://user-images.githubusercontent.com/91027877/212112189-b35c15f9-7d63-4386-8058-528381aacf79.gif)

6. Result

![image](https://user-images.githubusercontent.com/91027877/212114675-8dda2cdf-e8ac-437d-aab5-9aadcc8b1ce5.png)

