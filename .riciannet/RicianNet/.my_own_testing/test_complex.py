import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import caffe
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

mat=scipy.io.loadmat('../testdata/Brain1.mat')
noiseSigma=15
# mat['Img'] returns the same brain!
brain_not_normalized = mat['Img']

brain = brain_not_normalized / np.abs(brain_not_normalized).max()

brain_real=np.abs(brain)
# brain_real = (brain_real - brain_real.min()) / (brain_real.max() - brain_real.min())
# brain_real = brain_real / brain_real.max()


# plt.imsave('Brain1.png', np.fliplr(np.rot90(brain_real)), cmap='gray')
plt.imsave('Brain1.png', brain_real, cmap='gray')

# for noiseSigma in [5,10]:

level = noiseSigma * np.abs(brain).max() / 100
n1 = level * np.random.randn(*np.abs(brain).shape) + brain.real
n2 = level * np.random.randn(*np.abs(brain).shape) + brain.imag

in_img = np.sqrt(n1**2 + n2**2)
# in_img = (in_img - in_img.min()) / (in_img.max() - in_img.min())

plt.imsave(f'Contaminada_{noiseSigma}.png', in_img, cmap='gray')



prototext_path = '/mnt/data_drive/hrodrigo/mri_rician_noise/.riciannet/RicianNet/Riciannet_deploy.prototxt'
weights_path = '/mnt/data_drive/hrodrigo/mri_rician_noise/.riciannet/RicianNet/model/N15_complex_Brain1&2.caffemodel'


caffe.set_mode_cpu()  # Use caffe.set_mode_gpu() if you want to use GPU

net = caffe.Net(prototext_path,weights_path, caffe.TEST)

# print(dir(net))
# ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_backward', '_batch', '_blob_loss_weights', '_blob_names', '_blobs', '_bottom_ids', '_forward', '_inputs', '_layer_names', '_outputs', '_set_input_arrays', '_top_ids', 'after_backward', 'after_forward', 'backward', 'before_backward', 'before_forward', 'blob_loss_weights', 'blobs', 'bottom_names', 'clear_param_diffs', 'copy_from', 'forward', 'forward_all', 'forward_backward_all', 'inputs', 'layer_dict', 'layers', 'load_hdf5', 'outputs', 'params', 'reshape', 'save', 'save_hdf5', 'set_input_arrays', 'share_with', 'top_names']


# img_in = in_img.T

# np.rot90(np.rot90(np.rot90(img_in)))

result = net.forward(data=np.rot90(in_img)[None,None,...])
result_conv13=result['conv13'].squeeze()
# result_conv13 = (result_conv13 - result_conv13.min()) / (result_conv13.max() - result_conv13.min())

plt.imsave(f'Denoised_{noiseSigma}.png', result_conv13, cmap='gray')


plt.imsave(f'Brain1.png', np.rot90(brain_real), cmap='gray')

residuo=result_conv13-np.rot90(brain_real)
plt.imsave(f'Residuo_{noiseSigma}.png', residuo, cmap='gray')

print(psnr(result_conv13, np.rot90(brain_real),data_range=1.0))


error=(result_conv13.T - brain_real) ** 2
print(10 * np.log10(1 / error.mean()))


print(ssim(result_conv13, np.rot90(brain_real),data_range=1.0))


# array.reshape(1, 1, 256, 270)

# print(brain.dtype)
# print(in_img.dtype)

# TODO: 
    # ¿Cómo se calcularía la PSNR cuando la red ha limpiado 'más de la cuenta'? Es decir, hemos visto que las imágenes ya de por sí traen ruido.
    # Hacer normalización como figura en test_complex.m
    # ¿Quién es Brain1 en el contexto del repositorio de Brainweb-dl? Es una pregunta importante, ya que nppy segmenta toda la imagen volumétrica, devolviendo otra imagen volumétrica del cerebro.
print()
