# 👀 Pupil and Iris Localization Project

Welcome to the **Pupil and Iris Localization** project repository! This project aims to develop a **computer vision model** that accurately localizes the **pupil** and **iris** using **webcams**, with the ultimate goal of **understanding eye movements** and **gaze direction**. These insights will be used in **psychiatric diagnosis** scenarios, where subtle eye movement patterns can reveal key indicators about mental health conditions.

---

## 🚀 Overview

- **Objective**: Precisely detect the **pupil and iris** from webcam images.
- **Motivation**: 
  - Enable robust **eye-tracking** for **psychiatric assessments**, **diagnoses**, and **therapeutic** interventions.
  - Provide a **low-cost, non-invasive** solution using **standard webcams**.
- **Key Challenges**: 
  - Dealing with **lighting variations**, **occlusions** (eyelashes, eyelids, hair), **motion blur**, and **low-resolution** inputs from regular webcams.
  - Ensuring **real-time** or near real-time performance for practical usage in clinical settings.
  
---

## 🌐 Road Map

Below is a structured plan for the project:

1. **Investigate Current Papers** (✅ *Done*, but keeping the door open for more)
   - Gather insights from cutting-edge **iris and pupil localization** research.
   - Focus on **segmentation-free** models, **multi-task** learning frameworks, **CNN-based** approaches, and **dataset availability**.
     
2. **Deep Dive Reading** (🔎 *In Progress*)
   - Thoroughly review technical details, strengths, and gaps in these approaches.
   - Identify suitable **data augmentation**, **preprocessing**, and **model architectures**.

3. **Obtain Datasets** (🌐 *Next*)

| **Dataset Name**      | **Resolution** | **Format**        | **Sample Eye**    | **Link/Source**       |
|-----------------------|----------------|-------------------|-------------------|-----------------------|
| *Custom WebCam-Set*   | 640×480        | .jpg / .png       | *To be added*     | *TBD*                 |


4. **Explore Different Model Types** (🧐 *Research Approaches*)
   - Summarize potential **architectures** or strategies for pupil and iris localization:

| **Model Approach**                 | **Tried?** | **Notes**                                             |
|-----------------------------------|-----------|-------------------------------------------------------|
| **Object Detection** (e.g., YOLO) | ❌        | Could adapt bounding box for pupil/iris, might be less precise. |
| **Segmentation** (e.g., U-Net, YOLO)    | ❌        | Direct pixel-wise classification, common in iris tasks.  |
| **Keypoint Detection** (e.g., MMPose) | ❌        | Could predict pupil/iris boundary points.             |
| **Multi-Task** (e.g., multi-output CNN) | ❌ | Jointly learn pupil center, iris boundary, etc.        |
| **Others**                         | ❌        | ...e.g., ellipse fitting, classical edge-based, etc.    |

5. **Testing & Validation** (⚙️ *Implementation Stage*)

6. **Writing the Paper** (📝 *In Progress*)

7. **Merge & Present the Thesis** (🎉 *Final Stage*)
   - Combine all modules (datasets, models, results) into a **unified thesis**.
   - Prepare **slides**, **demos**, or any interactive **visualization** to showcase the system.

---

## 🏗️ Repository Structure (Example)

```
.
├── data/
│   ├── dataset1/ ...
│   ├── dataset2/ ...
├── docs/
│   └── references.md
├── models/
│   ├── segmentation/
│   ├── object_detection/
│   └── keypoint_detection/
├── scripts/
│   ├── data_preprocessing/
│   └── evaluation/
├── notebooks/
├── results/
└── README.md
```

---

## 📚 References

1. Webcam Eye Tracking.  
2. Segmentation-free Direct Iris Localization Networks.  
3. Iris Segmentation – Annotations.  
4. Accurate Pupil Center Detection in Off-the-Shelf Eye Tracking Systems Using Convolutional Neural Networks.  
5. A unified approach for automated segmentation of pupil and iris in on-axis images.  
6. A Ground Truth for Iris Segmentation.  
7. Current_Trends_in_Human_Pupil_Localization_A_Review.  
8. EyeDentify: A Dataset for Pupil Diameter Estimation based on Webcam Images.  
9. Joint Iris Segmentation and Localization Using Deep Multi-task Learning Framework.  
10. CASIA-Iris-Africa.  
11. Enhancing the Precision of Eye Tracking using Iris Feature Motion Vectors.  
12. EllSeg: An Ellipse Segmentation Framework for Robust Gaze.

---

> **Note**: This repository is a work in progress. Contributions, ideas, and suggestions are most welcome. Let’s build a robust, open-source solution for pupil and iris localization and push forward psychiatric diagnosis research! 


