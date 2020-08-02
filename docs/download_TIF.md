# Downloading TIF Assets and Files

Following this guide, you can learn how to download your generated irrigation map to a TIF asset on Google Earth Engine and to a TIF file on Google Drive.

## Download a TIF asset to your Google Earth Engine account

We can make use of our package's `write_image_asset` function for this.

First, train the model and generate irrigation predictions with `model.fit_predict()`. See the demo notebook `Demo.ipynb` for more instructions on how to do this.

Next, set the directory you want the asset to be stored in GEE using `set_asset_directory(dir)` like so:

```
# Set the directory you want your TIF asset to be stored in GEE
model.set_asset_directory("users/<GEE_USERNAME>/")
```

Next, call the `write_image_asset` function to save your asset to GEE with a specified asset name:

```
# Name of the TIF asset to be saved to GEE assets in base_asset_directory/image_asset_id
image_asset_id = "test_asset"

model.write_image_asset(image_asset_id)
```

Next, open up https://code.earthengine.google.com/ to check on the task as it's saving the TIF asset. You can see the task in the right column "Tasks". When the downloading task is complete, you will see the TIF asset in on the left column under the "Assets" tab.

![alt text](https://i.imgur.com/Pxf86yX.png)

## Download a TIF file to your Google Drive

We can make use of our package's `write_image_google_drive` function for this. Note there is no need to set a GEE base_asset_directory for this function.

After you've trained the model and generated irrigation predictions with `model.fit_predict()`, you can now call the `write_image_google_drive` like so:

```
# Name of the TIF file to be saved to your Google Drive
filename = "test_TIF"

model.write_image_google_drive(filename)
```

Note it may take a few minutes for the task to complete.

Next, open up http://drive.google.com/ to find the TIF file. Note this is saving to the Google Drive account associated with the Google account you used to register for GEE.
