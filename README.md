# Form_Recognition
# Form_Recognition

I have done the task of identifying the BOXES and identifying CHECKBOXEs using opencv in python 

To run the program to identify boxes in the image please run the following command.
python detect_shapes.py --boxes BOXES input_rectangle output_rectangle

To run the program to identify the checkboxes in the image please run the following command. 
python detect_shapes.py --checkboxes CHECKBOXES inputs output

For identifying the boxes I have used the technique of template matching. The Template folder will be required to identify the checkboxes. 
Template matching is not the best way to solve this problem. I would prefer using SHIFT to solve the problem. However, The SHIFT function is not available in the opencv 3 libraries due to patent constraints.
