<h2>Tensorflow-Image-Segmentation-Augmented-Breast-Ultrasound (2024/11/12)</h2>

This is an experiment of Image Segmentation for Breast-Ultrasound
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/1GcupOkETeymYR-WiGVW7i1Q7K1_94gAN/view?usp=sharing">
512x512 pixels Breast-Ultrasound-ImageMask-Dataset.zip</a>, which was derived by us from  
<a href="https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset">
Breast Ultrasound Images Dataset
</a>
<br>
<br>
<b>Data Augmentation Strategy:</b><br>
 To address the limited size of the Breast-Ultrasound dataset, 
 we employed <a href="./src/ImageMaskAugmentor.py">an online augmentation tool</a> to enhance segmentation accuracy, which supports the following aumentation methods.
<li>Horizontal flip</li>
<li>Rotation</li>
<li>Shrinks</li>
<li>Shears</li> 
<li>Deformation</li>
<li>Distortion</li>
<li>Barrel distortion</li>
<li>Pincushion distortion</li>
<br>

<br>
<hr>
<b>Actual Image Segmentation for Images</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/images/malignant (3).png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/masks/malignant (3)_mask.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test_output/malignant (3).jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/images/malignant (6).png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/masks/malignant (6)_mask.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test_output/malignant (6).jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/images/malignant (22).png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/masks/malignant (22)_mask.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test_output/malignant (22).jpg" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Breast-UltrasoundSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The dataset used here has been taken from the kaggle web-site
<a href="https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset">
Breast Ultrasound Images Dataset
</a>
<br>
<br>
<b>About Dataset</b><br>

Breast cancer is one of the most common causes of death among women worldwide. Early detection helps<br> 
in reducing the number of early deaths. The data reviews the medical images of breast cancer using <br>
ultrasound scan. Breast Ultrasound Dataset is categorized into three classes: normal, benign, and <br>
malignant images. Breast ultrasound images can produce great results in classification, detection, <br>
and segmentation of breast cancer when combined with machine learning.<br>
<br>

<b>Data</b><br>
The data collected at baseline include breast ultrasound images among women in ages between 25 and <br>
75 years old. This data was collected in 2018. The number of patients is 600 female patients. <br>
The dataset consists of 780 images with an average image size of 500*500 pixels. The images are in PNG format. <br>
The ground truth images are presented with original images. The images are categorized into three classes, <br>
which are normal, benign, and malignant.<br>
<br>
If you use this dataset, please cite:<br>
Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863.<br> 
DOI: 10.1016/j.dib.2019.104863.<br>
<br>
<b>License</b><br>
CC0: Public Domain

<br>
<br>
<h3>
<a id="2">
2 Breast-Ultrasound-ImageMask Dataset
</a>
</h3>
 If you would like to train this Breast-UltrasoundSegmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/1GcupOkETeymYR-WiGVW7i1Q7K1_94gAN/view?usp=sharing">
512x512 pixels Breast-Ultrasound-ImageMask-Dataset.zip</a>,
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Breast-Ultrasound
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>

<b>Breast-Ultrasound Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/Breast-Ultrasound_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
Therefore the online dataset augmentation strategy may be effective to improve segmentation performance.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained Breast-Ultrasound TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasoundand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0001
</pre>

<b>Online augmentation</b><br>
Enabled our online augmentation.  
<pre>
[model]
model         = "TensorflowUNet"
generator     = True
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>


<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/asset/epoch_change_infer.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 47  by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/asset/train_console_output_at_epoch_47.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Breast-Ultrasound.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/asset/evaluate_console_output_at_epoch_47.png" width="720" height="auto">
<br><br>Image-Segmentation-CDD-CESM-Breast-Ultrasound

<a href="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Breast-Ultrasound/test was not low, and dice_coef not high as shown below.
<br>
<pre>
loss,0.2537
dice_coef,0.6828
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Breast-Ultrasound.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/images/malignant (1).png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/masks/malignant (1)_mask.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test_output/malignant (1).jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/images/malignant (2).png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/masks/malignant (2)_mask.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test_output/malignant (2).jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/images/malignant (13).png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/masks/malignant (13)_mask.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test_output/malignant (13).jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/images/malignant (16).png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/masks/malignant (16)_mask.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test_output/malignant (16).jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/images/malignant (19).png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/masks/malignant (19)_mask.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test_output/malignant (19).jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/images/malignant (22).png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test/masks/malignant (22)_mask.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Ultrasound/mini_test_output/malignant (22).jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Breast Ultrasound Images Dataset</b><br> 
<a href="https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset">
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
</a><br>
Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. <br>
Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.<br>
<br>
