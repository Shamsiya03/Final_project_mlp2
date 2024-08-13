def run(name,namep):
    start_time0 = time.time()
    args = Args()
    pathbsd='./image\\'
    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # choose GPU:0
    input_image_path = pathbsd + name
    image = cv2.imread(input_image_path)

    softmax = nn.Softmax(dim=1)

    '''segmentation ML'''
    m = loadmat('./superpixel/'+namep+'.mat');
    seglab = m["seg_lab"]

    seg_map=seglab
    show = mark_boundaries(image, seg_map)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]

    '''train init'''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to(device)
    model = UNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
    image_flatten = image.reshape((-1, 3))
    color_avg = np.random.randint(255, size=(args.max_label_num, 3))
    show = image

    '''train loop'''
    start_time1 = time.time()
    model.train()
    for batch_idx in range(args.train_epoch):
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)[0]
        output1=output
        output1 = output1[np.newaxis, :, :, :]
        output2= output[0:1, :, :]
        croppings = (output2 > 0).float()

        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()

        '''refine'''
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]

        '''backward'''
        target = torch.from_numpy(im_target)
        target = target.to(device)
        loss = criterion(output, target) #as defined in Eq. (8)

        loss.backward()
        optimizer.step()

        '''show image''' #to print using the pesudo color;
        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        if un_label.shape[0] < args.max_label_num:  # update show
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(image.shape)
        cv2.imshow("seg_pt", show)
        cv2.waitKey(1)
        print(loss.item())
        if len(un_label) < args.min_label_num:
            break

    '''save'''
    sp=list(image.shape)
    label = im_target.reshape((sp[0],sp[1]))

    time0 = time.time() - start_time0
    time1 = time.time() - start_time1

    print('PyTorchInit: %.2f\nTimeUsed: %.2f' % (time0, time1))
    cv2.imwrite("output/seg_%s_%ds.jpg" % (namep, time1), show)
    return time1

