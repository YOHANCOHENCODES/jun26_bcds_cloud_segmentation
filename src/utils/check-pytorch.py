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

# NOTE by Uwe Rosner, 11.08.2026: On my Computer at home this outputs:
#
#    Is CUDA available? True
#    Number of GPUs: 1
#    Current GPU Name: NVIDIA GeForce RTX 5060
#    Tensor device: cuda:0 (Success!)
