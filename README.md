# Eco-Ex

Eco-Ex: a georeferenced dataset to train and test deep learning algorithms for economic exposure.

## Dataset Access

To acquire dataset, follow the instruction in ```Data_collection_guide.ipynb```

## Dataset Information

Datasets include economic exposure, remote sensing satellite imagery, night light intensity and road density

| Datasets                         | Description                                                  | Values                               | Format |
| -------------------------------- | ------------------------------------------------------------ | ------------------------------------ | ------ |
| economic exposure                | Property value, weighted by property size and property price | scaler, grid level and country level | csv    |
| remote sensing satellite imagery | Pictures of the Earth taken with remote sensing technology   | image,grid level                     | png    |
| night light intensity            | Night light information captured by remote sensing technology | scaler, grid level                   | csv    |
| road density                     | Road lengths, including road lengths, railway lengths and other road lengths | vector, grid level                   | csv    |

### Example Dataset

A sample of the dataset is provided at [sample data](data/used_data.csv) and image in [sample image](data/daytime_image/Fujian Province).

### Train

Run ```train_class.py``` to train grid level model, change *add_light* and *add_road* parameter to adjust the input of the model, outcomes are saved in [experiments](experiments)

Run  ```country_level_model.py```to train country level model, change *model_name* and *snapshot_name* to choose base model.

Run ```gradcam.py```to explain trained model, change *model_name* and *snapshot_name* to choose base model, set *i* to choose a picture.