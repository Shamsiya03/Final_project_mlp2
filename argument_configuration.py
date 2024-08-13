class Args(object):
    train_epoch =80 ## training iteration T ##
    mod_dim1 = 64  #
    mod_dim2 =100 #
    gpu_id =0 #0
    min_label_num = 4  # if the label number small than it, break loop
    max_label_num = 256  # if the label number small than it, start to show result image.