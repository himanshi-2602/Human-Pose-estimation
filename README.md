# Human-Pose Estimation



> **Abstract** – *This paper reports our experience with building a Human Pose Estimator. Our approach is based on Deep Learning and motivated by a paper titled [Multi-Person Pose Estimation](https://arxiv.org/pdf/1611.08050.pdf) by the Perceptual Computing Lab at Carnegie Mellon University. We have used the OpenCV DNN model and modified it to implement Tensorflow MobileNet Model and trained on the COCO dataset.*
> 

# Problem Statement

Human Pose Estimation is defined as the problem of localization of human joints (elbows, wrists, etc) in images or videos. It is also defined as the search for a specific pose in space of all articulated poses. Pose Estimation is a general problem in Computer Vision where we detect the position and orientation of an object. This usually means detecting key point locations that describe the object.

To analyze strong articulations, small and barely visible joints, and occlusions, and fight the challenge of clothing and lighting changes. 

![Sample Skeleton output of pose](https://github.com/ayushabrol13/Human-Pose-Estimation/raw/master/Human-Pose%20Estimation%20a98b239327d74cb48168a6c2ef3a6dd9/Untitled.png)

Sample Skeleton output of pose

## Why this problem is hard?

This is a challenging task because of the strong articulations, tiny, hardly perceptible joints, occlusions, clothes, and illumination variations.

## Literature Survey

We will Discuss two approaches used for Human pose estimation. 

- Classic approach
- Deep Learning based approach

Before 2014, the Classic approach was used until the introduction of the research paper, “Deep Pose” by Toshev et al which was based on Deep Learning.

| Classical Approach  | Deep Learning Based  |
| --- | --- |
| Represent an object by a collection of "parts" arranged in a deformable configuration | Pose estimation systems have universally adopted ConvNets as their main building block |
| Limitation of having a pose model not depending on image data | This strategy has yielded drastic improvements on standard benchmarks. |

### Advancements in Human Pose Estimation using Deep Neural Networks

We'll review a few studies that chart the development of human pose estimation in chronological order.

### ****[DeepPose: Human Pose Estimation via Deep Neural Networks](https://arxiv.org/pdf/1312.4659.pdf)****

- Deep Pose, was successful in beating existing models, in this approach, pose estimation is formulated as a CNN- based regression problem toward body joints.
- They use a cascade of such regressors to refine the pose estimates and get better estimates.
- This approach even works for a pose in a *holistic fashion,* i.e even if certain joints are hidden, they can be estimated if the pose is reasoned about holistically.
- The model consisted of an AlexNet backend (7 layers) with an extra final layer that outputs 2k joint coordinates - (xi,yi)∗2(xi,yi)∗2 for i∈{1,2…k}i∈{1,2…k} (where kk is the number of joints). The model is trained using an L2 loss for regression.

![Untitled](https://github.com/ayushabrol13/Human-Pose-Estimation/raw/master/Human-Pose%20Estimation%20a98b239327d74cb48168a6c2ef3a6dd9/Untitled%201.png)

- Regressing to XY locations is difficult and adds learning complexity which weakens generalization and hence performs poorly in certain regions.

### ****[Efficient Object Localization Using Convolutional Networks](https://arxiv.org/pdf/1411.4280.pdf)****

- This approach generates heatmaps by running an image through multiple resolution banks in parallel to simultaneously capture features at a variety of scales.
- The output is a discrete heatmap instead of continuous regression.
- A heatmap predicts the probability of the joint occurring at each pixel.
- A multi-resolution CNN architecture (coarse heatmap model) is used to implement a sliding window detector to produce a coarse heatmap output.
- They add an additional ‘pose refinement’ ConvNet that refines the localization result of the coarse heat map.
- The model consists of the heat-map-based parts model for coarse localization, a module to sample and crop the convolution features at a specified (x,y) location for each joint, as well as an additional convolutional model for fine-tuning.
- The model is trained by minimizing the Mean Squared-Error (MSE) distance of our predicted heat map to a target heat map (The target is a 2D Gaussian of constant variance (σ ≈ 1.5 pixels) centered at the ground-truth (x,y) joint location)

# Motivation

- Human Pose estimation is a very useful technique for various industry projects.
- It can be useful for a lot of detection applications, symptoms of knee injuries or spinal disorders can aid doctors in properly analyzing the posture and suggesting appropriate treatment.
- Nowadays, many applications are made to help during workouts, exercises, dance steps, etc. and the human pose estimation feature can help the users and the developers to create a better and more practical experience.
- Many people, especially the old age group practice yoga regularly to maintain a healthy lifestyle and can’t afford an instructor or classes, hence can use applications that can determine whether they’re on the right without the help of any other person.

# Study and Experiment settings

### Architecture Overview

![Untitled](https://github.com/ayushabrol13/Human-Pose-Estimation/raw/master/Human-Pose%20Estimation%20a98b239327d74cb48168a6c2ef3a6dd9/Untitled%202.png)

The model outputs the 2D positions of key points for each individual in the image after receiving as input a color image of size w*h. Three stages are involved in the detecting process:

1. **Stage 0:** The first 10 layers of the VGGNet are used to create feature maps for the input image.
2. ******************Stage 1:****************** A 2-branch multi-stage CNN is used where the first branch predicts a set of 2D confidence maps (S) of body part locations (e.g. elbow, knee, etc.).
    
    ![     Showing confidence maps for the Left Shoulder for the given image](https://github.com/ayushabrol13/Human-Pose-Estimation/raw/master/Human-Pose%20Estimation%20a98b239327d74cb48168a6c2ef3a6dd9/Untitled%203.png)
    
         Showing confidence maps for the Left Shoulder for the given image
    
3. ******************Stage 3:****************** The confidence and affinity maps are parsed by greedy inference to produce the 2D key points for all people in the image.
    
    ![Showing Part Affinity maps for Neck – Left Shoulder pair for the given image](https://github.com/ayushabrol13/Human-Pose-Estimation/raw/master/Human-Pose%20Estimation%20a98b239327d74cb48168a6c2ef3a6dd9/Untitled%204.png)
    
    Showing Part Affinity maps for Neck – Left Shoulder pair for the given image
    

### M****odels for Human Pose Estimation****

We used a Tensorflow **MobileNet Model** which is pre-trained on the [COCO](http://cocodataset.org/#keypoints-2018) dataset. The COCO model produces 18 points.

**COCO Output Format**
Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, LAnkle – 13, Right Eye – 14, Left Eye – 15, Right Ear – 16, Left Ear – 17, Background – 18 

## Demonstration

[Demonstration of the human pose estimator on a webcam. ](Human-Pose%20Estimation%20a98b239327d74cb48168a6c2ef3a6dd9/WhatsApp_Video_2022-11-10_at_8.04.33_PM.mp4)

Demonstration of the human pose estimator on a webcam. 

## Results and Output

Our Human Pose estimator will work for a .png **image**, mp4 **video file,** and even for the real-time **webcam**.

![Untitled](https://github.com/ayushabrol13/Human-Pose-Estimation/raw/master/Human-Pose%20Estimation%20a98b239327d74cb48168a6c2ef3a6dd9/Untitled%205.png)

![Untitled](https://github.com/ayushabrol13/Human-Pose-Estimation/raw/master/Human-Pose%20Estimation%20a98b239327d74cb48168a6c2ef3a6dd9/Untitled%206.png)

![Untitled](https://github.com/ayushabrol13/Human-Pose-Estimation/raw/master/Human-Pose%20Estimation%20a98b239327d74cb48168a6c2ef3a6dd9/Untitled%207.png)

## The web version of the model

### Demo

[UI Demo.mp4](Human-Pose%20Estimation%20a98b239327d74cb48168a6c2ef3a6dd9/UI_Demo.mp4)

## Future Work

### 3-D Pose Estimation

Three-dimensional (3D) human pose estimation involves estimating the articulated 3D joint locations of a human body from an image or video. A large number of approaches, with many based on deep learning, have been developed over the past decade, largely advancing the performance on existing benchmarks. 

Uses:

*Human-Computer Interaction*

A robot can better serve and help users if it can understand 3D poses, actions, and emotions of people. For example, a robot can take timely actions when it detects the 3D pose of a person who is prone to fall. They can interact with users in a better way.

*Self-driving cars*

They are required to make decisions to avoid collision with pedestrians, and thus understanding a pedestrian’s pose, movement and intention is very important

V*ideo surveillance*

It is of great significance for public safety. In this area, 3D pose estimation techniques could be used to assist in the re-identification task.

*Biomechanics and Medication*

Human movement can indicate the health status of humans. Thus, 3D pose estimation techniques could be used to construct a sitting posture correction system to monitor the status of users.

3-D pose estimation can be useful in the fashion industry, psychology, sports performance analysis, etc.

### Use of AI to reduce search time

We’ve 18 nodes as body parts, the output of which is a graph. We aim to apply search techniques to reduce search time. We can either use breadth-first or depth-first search-based algorithms to do the same. 

As we’ve only 18 nodes, we can explore both methods, in our opinion, a depth-first-based algorithm would give better accuracy, for example, let’s consider a yoga application where users can give an image or a video as input. 

As we’ll have only 18 nodes to explore, a DFS-based algorithm can explore one side of the human pose and predict whether the position is accurate and not search both sides, also it’ll backtrack as soon as it finds a ratio wrong and move on to the next image or ask the user to try again. 

### A* Algorithm

What is A* Algorithm? 

Commonly used to find the shortest path, the A* algorithm is a smart search algorithm used on trees and graphs. 

A* Search algorithms, unlike other traversal techniques, it has “brains”. What it means is that it is really a smart algorithm that separates it from the other conventional algorithms.

![Untitled](https://github.com/ayushabrol13/Human-Pose-Estimation/raw/master/Human-Pose%20Estimation%20a98b239327d74cb48168a6c2ef3a6dd9/Untitled%208.png)

 

Pseudo-code for the algorithm.

![Untitled](https://github.com/ayushabrol13/Human-Pose-Estimation/raw/master/Human-Pose%20Estimation%20a98b239327d74cb48168a6c2ef3a6dd9/Untitled%209.png)

**Heuristics**

**A)** **Exact Heuristics**

**B) Approximation Heuristics**

1. **Manhattan Distance**

![Untitled](https://github.com/ayushabrol13/Human-Pose-Estimation/raw/master/Human-Pose%20Estimation%20a98b239327d74cb48168a6c2ef3a6dd9/Untitled%2010.png)

1. **Diagonal Distance**

![Untitled](https://github.com/ayushabrol13/Human-Pose-Estimation/raw/master/Human-Pose%20Estimation%20a98b239327d74cb48168a6c2ef3a6dd9/Untitled%2011.png)

1. **Euclidean Distance**

![Untitled](https://github.com/ayushabrol13/Human-Pose-Estimation/raw/master/Human-Pose%20Estimation%20a98b239327d74cb48168a6c2ef3a6dd9/Untitled%2012.png)

## Conclusion

- We have used Deep Learning approach and implemented the OpenCV DNN model and trained the TensorFlow MobileNet model on the COCO dataset.
- For using our model for making an AI yoga instructor we have used the A* approach using DFS traversal.
- As A* is based on a Heuristic-based approach, so here each node will have a heuristic according to which Yoga pose we are doing. for example, If we are comparing the Tree Yoga pose, the node corresponding to the right knee will have the max heuristic as this joint is crucial in the Tree pose.

      

---

## References

1. [https://arxiv.org/pdf/1411.4280.pdf](https://arxiv.org/pdf/1411.4280.pdf)
2. [https://www.sciencedirect.com/science/article/pii/S1077314221000692#b131](https://www.sciencedirect.com/science/article/pii/S1077314221000692#b131)
3. [https://nanonets.com/blog/human-pose-estimation-2d-guide/#deeppose](https://nanonets.com/blog/human-pose-estimation-2d-guide/#deeppose)
4. [https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/](https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/)
5. [https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/#what-are-the-frameworks-that-it-supports](https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/#what-are-the-frameworks-that-it-supports)
6. [https://www.kdnuggets.com/2020/08/3d-human-pose-estimation-experiments-analysis.html](https://www.kdnuggets.com/2020/08/3d-human-pose-estimation-experiments-analysis.html)

###
