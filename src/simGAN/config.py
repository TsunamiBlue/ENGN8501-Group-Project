import torch

cuda_use = torch.cuda.is_available()
cuda_num = 0

init_lr = 0.001
delta = 0.5  # 0.5  # 0.0001

img_width = 55
img_height = 35
img_channels = 1

# synthetic image path
syn_path = 'data/syn/'
# real image path
real_path = 'data/real/'


# =================== training params ======================
# pre-train R times
g_pretrain = 1000
# pre-train D times
d_pretrain = 500
# train steps


batch_size = 128
# the history buffer size
buffer_size = 12800

k_d = 1  # number of discriminator updates per step
k_g = 50  # number of generative network updates per step, the author of the paper said it's 50

# output R pre-training result per times
r_pre_per = 50
# output D pre-training result per times
d_pre_per = 10
# output formal training result per times
f_per = 1
# save model dictionary and training dataset output result per train times
save_per = 1000

# file root
save_path = 'models'

# dictionary saving path
D_path = 'D_%d.pkl'
R_path = 'R_%d.pkl'

optimizer_path = 'optimizer_status.pkl'