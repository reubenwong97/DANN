python experiment_selfens_meanteacher.py --exp=visda18_train_val_red --arch=resnet152 --img_size=160 --batch_size=56 --double_softmax --use_dropout --src_hflip --tgt_hflip --src_affine_std=0.0 --tgt_affine_std=0.0 --src_rot_std=0.0 --tgt_rot_std=0.0 --src_scale_u_range=0.75:1.333 --tgt_scale_u_range=0.75:1.333 --src_colour_rot_std=0.0 --tgt_colour_rot_std=0.0 --src_colour_off_std=0.0 --tgt_colour_off_std=0.0 --img_pad_width=16 --epoch_size=target --unsup_weight=10.0 --cls_balance=0.01 --confidence_thresh=0.9 --fix_layers=conv1,bn1,layer1 --num_epochs=3 --learning_rate=1e-5 --log_file=results_visda18/res_visda18_traintest_resnet152_basicaug_run_MINI.txt --result_file=results_visda18/history_visda18_traintest_resnet152_basicaug_run_MINI.h5 --model_file=results_visda18/model_visda18_traintest_resnet152_basicaug_run_MINI.pkl