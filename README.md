[![HitCount](http://hits.dwyl.io/srihari-humbarwadi/cartoon_gan_tf20.svg)](http://hits.dwyl.io/srihari-humbarwadi/cartoon_gan_tf20)

## Implementation details
 - Dataset : cartoon100k
 - batch_size : 64
 - H, W = [256, 256]
 - Discrminator_lr = 0.00025
 - Generator_lr = 0.0002
 - Weights init = glorot_uniform (to be changed to RandomNormal[mean=0, stddev=0.02])
 - Discriminator training strategy : separate weight updates for real and fake batch
 

 ## Outputs
<a href="outputs/boy_boy_dark_glasses.gif" target="_blank"><img 
src="outputs/boy_boy_dark_glasses.gif" alt="boy_boy_dark_glasses.gif" width="285" height="285" 
border="10" /></a>
<a href="outputs/boy_boy_short_hair.gif" target="_blank"><img 
src="outputs/boy_boy_short_hair.gif" alt="boy_boy_short_hair.gif" width="285" height="285" 
border="10" /></a>
<a href="outputs/boy_girl.gif" target="_blank"><img 
src="outputs/boy_girl.gif" alt="boy_girl.gif" width="285" height="285" 
border="10" /></a>

<a href="outputs/boy_boy.gif" target="_blank"><img 
src="outputs/boy_boy.gif" alt="boy_boy.gif" width="285" height="285" 
border="10" /></a>
<a href="outputs/boy_girl_dark_glasses.gif" target="_blank"><img 
src="outputs/boy_girl_dark_glasses.gif" alt="boy_girl_dark_glasses.gif" width="285" height="285" 
border="10" /></a>
<a href="outputs/boy_girl_short_hair.gif" target="_blank"><img 
src="outputs/boy_girl_short_hair.gif" alt="boy_girl_short_hair.gif" width="285" height="285" 
border="10" /></a>
