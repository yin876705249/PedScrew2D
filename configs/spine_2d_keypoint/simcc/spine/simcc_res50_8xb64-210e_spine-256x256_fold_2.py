_base_ = ['./simcc_res50_8xb64-210e_spine-256x256_base.py']

train_dataloader = _base_.train_dataloader.copy()
val_dataloader = _base_.val_dataloader.copy()
test_dataloader = _base_.test_dataloader.copy()

train_dataloader['dataset']['ann_file'] = 'annotations/fold_2/spine_2d_train.json'
val_dataloader['dataset']['ann_file'] = 'annotations/fold_2/spine_2d_val.json'
test_dataloader['dataset']['ann_file'] = 'annotations/fold_2/spine_2d_test.json'