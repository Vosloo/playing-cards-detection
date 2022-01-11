<center> <h1> Playing cards dataset creator </h1> </center>

<b>This README is still in progress. Please contact the authors in case of any problems!</b>

All you need to have are the scans of your cards in the images/scans/data/input directory.

Then run the project.ipynb file to extract playing cards from the scans and prepare backgrounds.

Then in a section of the project.ipynb file "or run do it in a batch" (to be changed) you'll have to manually label the cards corresponding to the images marked as a ROI_{number}.png in a images/scans/data/output directory.After that, you can run the project.ipynb file again to end the extraction.

After than run the dataset_creator.py file with the parameters of your choice (size of the dataset etc.) to create the dataset.

Final step is to run the split_data.py file to split the data into three sepearate sets: training, validation and test.