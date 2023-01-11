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





