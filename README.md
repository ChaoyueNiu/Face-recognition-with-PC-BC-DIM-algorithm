The main program is to resize and align face images so that PC/BC-DIM algorithm can be tested on ORL face datasets.

In the root directory, run the command classify_images ('ORL', 31) for images with the size of 92 by 112, 117 by 135, 64 by 64 and so on. After then, 21 figures will be obtained. By changing the different datasets of att_faces, we can get different results under the condition of different size of images within the dataset.

Open-source software, written in MATLAB, which performs all the experiments described in this article is available for download from: http://www.corinet.org/mike/Code/pcbc_image_recognition.zip.

ORL face dataset is also avaliable for download from: http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

This code is modified based on 3.3.1 program and 3.2.6 program, I mainly focus on modofying demo.m program to develop an automatic method by combining 3.3.1 with 3.2.6.
if you want to change size of image, please open my_alignment_2points_demo.m to modify original dataset is in att_faces folder
Running demo.m , it will generate images resized and aligned, which will be saved in root dictionary.
