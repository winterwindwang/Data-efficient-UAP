# Data-efficient-UAP

The official implementation code of [**Improving Transferability of Universal Adversarial Perturbation with Feature Disruption**](http://doi-org-s.webvpn.zju.edu.cn:8001/10.1109/TIP.2023.3345136), which had been accepted in *IEEE Transaction on Image Processing*. The table of contents is as follows.

We upload some baseline UAPs in [here](https://github.com/winterwindwang/Awesome-UAPs.git).

# Category
- [Settings](#settings)
- [Train](#training)
- [Test](#testing)
- [Downstream](#ds)

# <a id=settings></a>1. Settings: preparing training and test dataset

First, you should prepare the dataset (i.e., ImageNet) as following folder structure:

## 1.1 ImageNet & Downstream dataset
These two types of datasets should be arranged by each image placed in their category folder. News: we release the constructed [ImageNet10k](https://github.com/winterwindwang/Data-efficient-UAP/blob/main/dataset/imagenet10k.txt), we train our UAP on 2000 images randomly sampled from this. 
```python
-- ImageNet_folder
    -- train_folder
    	-- n01440764
        -- n01443537
        -- ...
        -- n15075141
    -- val_folder
    	-- n01440764
        -- n01443537
        -- ...
        -- n15075141
```
## 1.2 COCO & VOC & Mixed
These three types of datasets should be arranged by the image folder directly, as our method does not require the label. These three types of datasets should be arranged by the image folder directly, as our method does not require the label. Note that the mixed is composed of four datasets, so they should be assigned with a list of datapaths according to this dataset format.  
```python
-- COCO2014
  -- train2014
    -- image_01.jpg
    -- xxx
```

# <a id=training></a>2. Training: train the UAP for different models
We provide the code and the corresponding description for the train the UAP

```python
python main.py  \
    --exp_name "exp" \
    --model_name "resnet50" \
    --train_data_name "imagenet" \
    --test_data_name "imagenet" \
    --train_data_path "train data path" \
    --test_data_path "test data path" \
    --save_dir "checkpoints" \
    --eps 10. \
    --miu 0.1 \
    --eps_step 0.001 \
    --input_size 224 \
    --batch_size 50 \
    --epochs 50 \
    --num_works 8 \
    --nb_images 2000 \
    --feat_type "half_feat_and_grad" \
    --loss_type "abs" \
    --sort_type "channel_mean" \
``` 
Description
```python
python main.py  \
    --exp_name "exp" \  # folder name to save the intermediate result
    --model_name "resnet50" \ # the victim model
    --train_data_name "imagenet" \  # the dataset used to train the UAP 
    --test_data_name "imagenet" \  # the dataset used to test the UAP
    --train_data_path "train data path" \  # the data path used to train the UAP
    --test_data_path "test data path" \  # the data path used to test the UAP
    --save_dir "checkpoints" \    # folder used to save the final trained results
    --eps 10. \  # the maximum modification of the UAP
    --miu 0.1 \  # the weight of momentum
    --eps_step 0.001 \  # the updation step
    --input_size 224 \  # image size
    --batch_size 50 \   # batch size
    --epochs 50 \      # max training epoch
    --num_works 8 \   
    --nb_images 2000 \  # number of training image
    --feat_type "half_feat_and_grad" \  # feature usage type
    --loss_type "abs" \    # loss function design type
    --sort_type "channel_mean" \  # feature importance calculation standard  
``` 

# <a id=testing></a>3. Testing: test the trained UAP
During test stage, you need to assign the ImageNet validation path and the UAP files, and can get the evaluation result.

We provided our trained UAP files in:

| Different in Training settings | Link |
|----|----|
| Model | [here](perturbations/) |
| Number of training samples | [here]() |
| Dataset | [here]() |

Download or assign the UAP file in the uap_validation_imagenet.py 

Run the following code to get the final result.

```bash
python uap_validation_imagenet.py 
```
Note that, the evaluation of UAP on downstream task can be achieved by assigning the corresponding dataset folder. The code is as follows
```python
python uap_validation_imagenet.py --is_downstream True
```


# <a id=ds></a>4. Downstream Tasks
We provide three manner to train the downstream task
+ From Scratch: We do not evalute the UAP's performance on these models 
+ Fixed: Fixed the feature extractor and finetuned the classification head
+ Fullnet: Finetuned the feature extraction and classification head

Training detail refers to `transfer_learning.py`, using the follow code to perform training
```python
python transfer_learning.py 
```
