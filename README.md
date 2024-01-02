# Data-efficient-UAP

The official implementation code of [**Improving Transferability of Universal Adversarial Perturbation with Feature Disruption**](http://doi-org-s.webvpn.zju.edu.cn:8001/10.1109/TIP.2023.3345136), which had been accepted in *IEEE Transaction on Image Processing*.

We provide the necessary steps to reproduce or test our work.


## 1. Preparing trainging and test dataset

First, you should prepare the dataset (i.e., ImageNet) as following folder structure:

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

## 2. Training the UAP for the model
### TODO
## 3. Testing our provided UAPs.
During test stage, you need to assign the ImageNet validation path and the UAP files, and can get the evaluation result.

We provided our trained UAP files in:

| Different in Training settings | Link |
|----|----|
| Model | [here]() |
| Number of trianing samples | [here]() |
| Dataset | [here]() |

Run the following code to get the final result.

```bash
python test_uaps.py --data_dir "ImageNet_folder"
```



