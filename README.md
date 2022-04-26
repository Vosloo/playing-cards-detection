<center> <h1> Playing cards dataset creator </h1> </center>

<h2> About </h2>

There exists datasets with specific sets of playing cards but they are usually not suitable for more specifically-looking sets. Facing such problem, the repository was created so that the process of creating dataset based on arbitrary set of playing cards would be an easy process.

This repository allows for easily create big datasets of playing cards for card classification purposes. The created datasets consists of random background (to help with generalization) and one, two or three cards stacked on each other (with additional transformations such as tilt and rotation). The created dataset was used in another repository: [game of Russian Schnapsen](https://github.com/Vosloo/russian-schnapsen) with very good accuracy of playing card detection.

<h2> How to run </h2>

The process of creating the dataset will be made easier but for now steps are:

 - You have to prepare (good quality) scans of your entire set of playing cards and put in into images/scans/data/input directory.
 - Run the project.ipynb file to extract playing cards from the scans and prepare backgrounds.
 - Then in a section of the project.ipynb file "or run do it in a batch" (to be changed) you'll have to manually label the cards corresponding to the images marked as a ROI_{number}.png in a images/scans/data/output directory.After that, you can run the project.ipynb file again to end the extraction. 
 - After than run the dataset_creator.py file with the parameters of your choice (size of the dataset etc.) to create the dataset.
 - Final step is to run the split_data.py file to split the data into three sepearate sets: training, validation and test for classification purposes of YOLOv5

<h2> Limitations </h2>

Images created for the dataset does not contain different lightning levels which would increase generalization even more. One could extend the creation of images artificially adding different lightning level after processing the final image.
