 #### DIRECTORY ORDER

 16_channel/
        -- data/
             -- sixteen_band
             -- three_band
             -- grid_sizes.csv
             -- sample_submission.csv
             -- train_wkt_v4.csv
        -- scripts/
             --
        -- cache


### TRAINING
--- first run the file 'get_3_band_shapes.py' and this will create '3_shapes.csv' in data folder
--- run the  'cache_train.py'
--- now run the 'rgb_model.py'

### TESTING
--- after running 'get_3_band_shapes.py' and 'cache_train.py' , run 'rgb_prediction.py'

