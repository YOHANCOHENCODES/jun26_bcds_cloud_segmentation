import torch

cuda_available = torch.cuda.is_available()
print(f'Is CUDA available? {cuda_available}')

if cuda_available:
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    print(f'Current GPU Name: {torch.cuda.get_device_name(0)}')
    x = torch.rand(3, 3).cuda()
    print(f'Tensor device: {x.device} (Success!)')
else:
    print('PyTorch is currently using the CPU.')
