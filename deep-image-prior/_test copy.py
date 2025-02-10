from _utils import *
import torch

img = load_gray_image("dali-bw.jpg")[None, ...]  # add batch dimension

# funs = [grads]
# for fun in funs:
#     results = fun(img)
#     img_grad = torch.zeros_like(results[0])
#     for i in range(len(results)):
#         print_image(
#             (results[i].squeeze().numpy() * 255).astype(np.uint8),
#             f"{fun.__name__}_{i}.png",
#         )
#         img_grad += results[i] ** 2
#     print_image(
#         (img_grad.sqrt().squeeze().numpy() * 255).astype(np.uint8),
#         f"{fun.__name__}_abs.png",
#     )

# results = roberts(img)
# img_grad = torch.zeros_like(results[0])
# for i in range(len(results)):
#     print_image(
#         (results[i].squeeze().numpy() * 255).astype(np.uint8), f"roberts_{i}.png"
#     )
#     img_grad += results[i] ** 2
# print_image(
#     (img_grad.sqrt().squeeze().numpy() * 255).astype(np.uint8), "roberts_abs.png"
# )

# results = prewitt(img)
# img_grad = torch.zeros_like(results[0])
# for i in range(len(results)):
#     print_image(
#         (results[i].squeeze().numpy() * 255).astype(np.uint8), f"prewitt_{i}.png"
#     )
#     img_grad += results[i] ** 2
# print_image(
#     (img_grad.sqrt().squeeze().numpy() * 255).astype(np.uint8), "prewitt_abs.png"
# )

# results = kirsch(img)
# img_grad = torch.zeros_like(results[0])
# for i in range(4):
#     print_image(
#         (results[i].squeeze().numpy() * 255).astype(np.uint8), f"kirsch_{i}.png"
#     )
#     img_grad += results[i] ** 2
# print_image(
#     (img_grad.sqrt().squeeze().numpy() * 255).astype(np.uint8), "kirsch_abs.png"
# )


results = derivative5(img)
img_grad = torch.zeros_like(results[0])
for i in range(len(results)):
    print_image(
        (results[i].squeeze().numpy() * 255).astype(np.uint8),
        f"derivative5_{i}.png",
    )
    img_grad += results[i] ** 2
print_image(
    (img_grad.sqrt().squeeze().numpy() * 255).astype(np.uint8), "derivative5_abs.png"
)
