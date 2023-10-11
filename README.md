# Human-Action-Recognition-In-The-Dark

This a project based on the ARID dataset, which focuses on Human Activity Detection in Low-light Conditions. Here we are using ResNet-18 to train the model and use openpose heatmap to detect the joints of human. 
For furthur details follow the slide below. 
https://docs.google.com/presentation/d/14TbkgY263BwlU7PCzEROL7dAjucNPDvDx67kZTB4Ejc/edit?usp=sharing

1. Install environment using env.yaml

2. Copy the dataset to the main folder (Human-Action-Recognition-In-The-Dark) and rename it as EE6222_data. The EE6222_data structure   should be as follow:
    Human-Action-Recognition-In-The-Dark
    └───EE6222_data
        ├───train
        ├───validate
        ├───mapping_table.txt
        ├───train.txt
        └───validate.txt
3. Run the main_inference.py to obtain the test result (TR). The test result (TR) can be obtained at the Submission folder.

4. See the PDF report Action Recognition in the Dark for detailed descriptions.
