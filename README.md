# SBGAR

This repository is the code of [SBGAR: Semantics based group activity recognition](https://openaccess.thecvf.com/content_iccv_2017/html/Li_SBGAR_Semantics_Based_ICCV_2017_paper.html).

___

Considering this is an old project, it is not easy to clean up the whole project. We simpliy our code and release the caption generation module. 

Specifically, we modified the code by **removing the part of extracting CNN feature from optical flow images**, so that one can download the Volleyball dataset and run the code directly (no need to generate optical flow images).

After getting the generated image captions, one can easily predict human activity labels based on these captions using a CNN model (here is a [demo code](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)). 
 
To run the code:
1. Download Volleyball dataset and unzip it. (path: <Volleyball_data_path>) 
2. Set the first parameter, namely "dataset_root_path", in file "Configuration.py" to <Volleyball_data_path>.
3. Download pre-trained inception-v3 from http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz, unzip the file and copy "classify_image_graph_def.pb" to our "models" folder.
4. in “code” folder, run “python main.py"
 
P.S.: the code was implemented based on Python 2.7 & Tensorflow 0.12.



