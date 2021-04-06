To run:

1) Download dataset train + validation visda18 into folder 'dataset'. 
   02 .txt files are preloaded in the folder for your reference.
  ('image_list.txt' lists the full dataset (original from visda18 website; 
   'image_list_red.txt' lists a small part of dataset. I use image_list_red.txt for debugging purpose cuz execution is faster)

2) Check 2 files:
   + visda18_dataset.py (Full dataset run)
     Makes sure   datafile is pointed to 'image_list.txt'

        train_dir = settings.get_data_dir('visda18_clf_train')
        file_list_path = os.path.join(train_dir, 'image_list.txt')


        val_dir = settings.get_data_dir('visda18_clf_validation')
        file_list_path = os.path.join(val_dir, 'image_list.txt')


   + visda18_dataset_red.py (Mini dataset run)
     Makes sure   datafile is pointed to 'image_list_red.txt'

        train_dir = settings.get_data_dir('visda18_clf_train')
        file_list_path = os.path.join(train_dir, 'image_list.txt')


        val_dir = settings.get_data_dir('visda18_clf_validation')
        file_list_path = os.path.join(val_dir, 'image_list.txt')


3) Train + test:
  = Full dataset run: Run code in run_visda18_trainval_resnet152_basicaug.sh
  = Mini dataset run: Run code in run_visda18_trainval_resnet152_basicaug_MINI.sh (faster but not actual results, for debugging) [This option takes me around 20 mins in my local machine]
4) Results will be in .h5 and .txt log in results_visda18