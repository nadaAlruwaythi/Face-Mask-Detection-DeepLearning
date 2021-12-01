# Face Mask Detection 

### Minimum Viable Product

</br>

#### Goal of the Project
The purpose of this project is to build a deep learning model that recognizes persons wearing masks, people with no masks, and people wearing erroneous masks.

#### EDA
- Datatset: Found 8982 files belonging to 3 classes.
- Three classes : {'with_mask': 0, 'without_mask': 1, 'mask_weared_incorrect': 2}

<img src="https://github.com/Wafaa-Alharbi/Face-Mask-Detection-DeepLearning/blob/main/Images/sample.PNG" width="700"/> <br/>


</br>


- rescale the images size to 128 x 128

- distribution of the target variable was a fairly uniform distribution , So there are no class imbalance issue.

<img src="https://github.com/Wafaa-Alharbi/Face-Mask-Detection-DeepLearning/blob/main/Images/Distribution_of_classes.PNG" width="700"/> <br/>




#### Next Steps
1. improve accuracy scores
2. Using Transfer Learning (VGG16)
3. Image augmentation (increase training samples by rotating and horizontal flipping images)
4. Apply model on steamlit app (if I have enough time)
