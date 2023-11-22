def dice_loss(y_true, y_pred, smooth=1e-7):
    y_true_f = paddle.flatten(y_true,stop_axis=-2)[:,0:5]
    y_pred_f = paddle.flatten(y_pred,stop_axis=-2)[:,0:5]
    intersect = paddle.sum(y_true_f * y_pred_f, axis=0)
    denom = paddle.sum((y_true_f + y_pred_f), axis=0)
    return 1-paddle.mean((2. * intersect + smooth) / (denom + smooth))

import random

class RandConv_source(nn.Layer):
    def __init__(self,kernel_size):
        super(RandConv_source,self).__init__()
        self.relu = nn.ReLU()
        self.down1 = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=int(kernel_size[0]), padding='same')
        self.down2 = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=int(kernel_size[1]), padding='same') 
      
    def forward(self,inputs):
        x = self.down1(inputs)
        x = self.relu(x)
        x = self.down2(x)
        return x

def get_source_trans(images_source):
    rand_conv_kernel_lists=np.array([1,3,5])
    rand_conv=RandConv_source(kernel_size=np.random.choice(rand_conv_kernel_lists,size=2))
    coe_mix = random.random()
    images_source_trans_final = ( coe_mix*(rand_conv(images_source)) + (1-coe_mix)*images_source ) / 2
    return images_source_trans_final

def foreground_mixup(images_source,images_source_trans_final,labels_source):

    for b in range(images_source.shape[0]):
      if(random.random() > 0.9):
        for ii in range(16):
            for jj in range(16):
                
                if( ( np.unique( labels_source[b, ii*16 : (ii+1)*16 , jj*16 : (jj+1)*16] ).any() ) != 0):
                    
                        temp = images_source[b,0, ii*16 : (ii+1)*16 , jj*16 : (jj+1)*16]
                        images_source[b,0, ii*16 : (ii+1)*16 , jj*16 : (jj+1)*16] = images_source_trans_final[b,0, ii*16 : (ii+1)*16 , jj*16 : (jj+1)*16]
                        images_source_trans_final[b,0, ii*16 : (ii+1)*16 , jj*16 : (jj+1)*16] = temp
                      
    images_source.stop_gradient = True
    images_source_trans_final.stop_gradient = True
    return images_source,images_source_trans_final
    
def train(path,epochs,lr):

    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr,T_max=9500), warmup_steps=500, start_lr=0.000001, end_lr=lr)
    optim_segmentor = paddle.optimizer.Adam(learning_rate=scheduler, parameters=segmentor.parameters())
    optim_segmentor_feature = paddle.optimizer.Adam(learning_rate=scheduler, parameters=segmentor.parameters()[:82])


    for epoch in range(epochs):
        segmentor.train()

        losses_contrastive=[]
        losses_ct_consistency=[]
        losses_segment=[]


        for j in range(0,100,1):
            
            batch_size = 8
        
            #segmentation for source
            images_source,labels_source,dic=functions.pick_data(path='data/mmwhs/gt_train_{}/'.format(source_name),training=1,num_slices=batch_size,num_foreground_slices=batch_size//4)
            images_source_trans_final=get_source_trans(images_source)
            images_source,images_source_trans_final = foreground_mixup(images_source,images_source_trans_final,labels_source)
  
            segment_mr0=segmentor(images_source,mode='pure')
            segment_mr1=segmentor(images_source_trans_final,mode='pure')

            loss_segment00=dice_loss(y_pred=nn.functional.softmax(paddle.transpose(segment_mr0,perm=[0,2,3,1]),axis=-1),y_true=nn.functional.one_hot(labels_source,num_classes=5))
            loss_segment10=dice_loss(y_pred=nn.functional.softmax(paddle.transpose(segment_mr1,perm=[0,2,3,1]),axis=-1),y_true=nn.functional.one_hot(labels_source,num_classes=5))
            loss_segment0=(loss_segment00+loss_segment10)/2
        
            loss_segment01=nn.functional.cross_entropy(input=paddle.transpose(segment_mr0,perm=[0,2,3,1]),label=labels_source)
            loss_segment11=nn.functional.cross_entropy(input=paddle.transpose(segment_mr1,perm=[0,2,3,1]),label=labels_source)
            loss_segment1=(loss_segment01+loss_segment11)/2

            loss_segment=(loss_segment00+loss_segment01)/2
            losses_segment.append(loss_segment.item())

            optim_segmentor.clear_grad()
            loss_segment.backward()
            optim_segmentor.step()

            #segmentation for target
            images_target,sda_images_target,_=functions.sda_pick_data(path='data/mmwhs/gt_train_{}/'.format(target_name),num_slices=batch_size)

            segment_cons0=segmentor(images_target,mode='pure')
            segment_cons1=segmentor(sda_images_target,mode='pure')

            loss_ct_consistency = nn.functional.mse_loss(input=segment_cons0,label=segment_cons1)
            losses_ct_consistency.append(loss_ct_consistency.item())
            
            optim_segmentor_feature.clear_grad()
            loss_ct_consistency.backward()
            optim_segmentor_feature.step()
         

            scheduler.step()

        '''if(epoch%50==0 and epoch!=0):

            paddle.save(segmentor.state_dict(),'model/segmentor{}'.format(epoch))
            paddle.save(discriminator_feature.state_dict(),'model/discriminator_feature{}'.format(epoch))
            [paddle.save(mlp_list_seg[i].state_dict(),'model/mlp_seg{}_{}'.format(i,epoch)) for i in range(5)]'''


        print('epoch:{} consistency:{} losses_contrastive:{} segment:{}'.format(
            epoch,
            np.mean(losses_ct_consistency),
            np.mean(losses_contrastive),
            np.mean(losses_segment)
            ))
