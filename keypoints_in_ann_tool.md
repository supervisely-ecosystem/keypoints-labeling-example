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
