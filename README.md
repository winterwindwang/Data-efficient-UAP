# Data-efficient-UAP

The official implementation code. We provide the UAP trained on different network, and we offer the validation code for evaluate the our UAPs. The overall implementation code will be released soon!

## Usage

You only need to assign the ImageNet validation path, and can get the evaluation result.

The data folder should be arranged as follows

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

Run the following code

```bash
python test_uaps.py --data_dir "ImageNet_folder"
```



