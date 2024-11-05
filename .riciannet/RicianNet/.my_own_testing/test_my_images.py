import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import caffe
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image


u = np.array(Image.open('Original_Mine.png').convert('L')) 
u = u /u.max()
# plt.imsave('Original_Mine.png', u, cmap='gray')

noise_factor = .15
sigma=noise_factor * u.max()

f_real = u + sigma * np.random.randn(u.shape[0], u.shape[1])
f_imag = sigma * np.random.randn(u.shape[0], u.shape[1])
f = np.sqrt(f_real**2 + f_imag**2)
residue = f-u

plt.imsave(f'Contamined_{noise_factor}_Mine.png', f, cmap='gray')

plt.imsave(f'Residue_{noise_factor}_Mine.png', residue, cmap='gray')

prototext_path = '/mnt/data_drive/hrodrigo/mri_rician_noise/.riciannet/RicianNet/Riciannet_deploy.prototxt'
weights_path = '/mnt/data_drive/hrodrigo/mri_rician_noise/.riciannet/RicianNet/model/N15_complex_Brain1&2.caffemodel'

caffe.set_mode_cpu() 
net = caffe.Net(prototext_path, weights_path, caffe.TEST)

result = net.forward(data=f.reshape(1, 1, 256, 270))
result_conv13=result['conv13'].reshape(256,270)
plt.imsave(f'Denoised_{noise_factor}_Mine.png', result_conv13, cmap='gray')

