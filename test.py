import medpy.metric.binary as mmb
@paddle.no_grad()
def test_dice_final(segmentor,path):
    segmentor.eval()
    filenames=os.listdir(path)
    dice_list=np.array([0.,0.,0.,0.,0.])
    assd_list=np.array([0.,0.,0.,0.,0.])

    for f in filenames:
        data=np.load(path+f)

        voxel = data['arr_0'].astype('float32')
        voxel = np.flip(voxel, axis=0)
        voxel = np.flip(voxel, axis=1)
        label = data['arr_1'].astype('int32')
        label = np.flip(label, axis=0)
        label = np.flip(label, axis=1)
        label = np.transpose(label,axes=[2,0,1])

        if('ct' in path):
            voxel = np.subtract(np.multiply(np.divide(np.subtract(voxel, -2.8), np.subtract(3.2, -2.8)), 2.0), 1)
            batch_size=32      #256/8=32
        if('mr' in path):
            voxel = np.subtract(np.multiply(np.divide(np.subtract(voxel, -1.8), np.subtract(4.4, -1.8)), 2.0),1)
            batch_size=26      #130/5=26

        for i in range(0,voxel.shape[-1],batch_size):
            image=[]
            for j in range(batch_size):
                slice_t = voxel[...,i+j]
                slice_t = np.expand_dims(slice_t,0)
                image.append(slice_t)
            image = paddle.to_tensor(image)
            predict=segmentor(image)
            predict=paddle.argmax(predict,axis=1)

            if(i==0):
                predicts=predict
            else:
                predicts=paddle.concat([predicts,predict],axis=0)
        for c in range(1,5,1):
            predicts_temp=np.copy(predicts.numpy())
            label_temp=np.copy(label)

            predicts_temp[predicts_temp!=c]=0
            label_temp[label_temp!=c]=0
            dc=mmb.dc(predicts_temp, label_temp)
            
            try:
                asd1 = mmb.assd(predicts_temp, label_temp)
            except:
                asd1 = 1
            assd_list[c-1]+=asd1
            dice_list[c-1]+=dc
    dice_list[4]=np.mean(dice_list[:4])
    assd_list[4]=np.mean(assd_list[:4])
    dice_list=100*dice_list/4
    assd_list=assd_list/4
    return dice_list[4],assd_list[4]

test_dice_final(segmentor,path) #path = test data location
