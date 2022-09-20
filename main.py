import argparse
import Dataloader as data
import torch
import time
import os
import vae
import tooltip_plot as tooltip

def str2bool(v):
    """
    Str to Bool converter for wrapper script.
    This is used both for --from_ckpt flag, which
    is False by default but can be turned on either by listing the flag (without args)
    or by listing with an appropriate arg (which can be converted to a corresponding boolean)
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


"""# args
parser = argparse.ArgumentParser(description="User data for VAE")

parser.add_argument('--data_dir', type=str, metavar='N', default='',required=True, \
help='Path to directory where data lives.')
parser.add_argument('--save_dir', type=str, metavar='N', default='', \
help='Dir where model params, latent projection maps and TB logs are saved to. Default is to save files to current dir.')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', \
help='Input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',\
help='Number of epochs to train (default: 100)')
parser.add_argument('--seed', type=int, default=1, metavar='S', \
help='Random seed (default: 1)')
parser.add_argument('--save_freq', type=int, default=10, metavar='N', \
help='How many epochs to wait before saving training status.')
parser.add_argument('--test_freq', type=int, default=2, metavar='N', \
help='How many epochs to wait before testing.')
parser.add_argument('--from_ckpt', type=str2bool, nargs='?', const=True, default=False, \
help='Boolean flag indicating if training and/or reconstruction should be carried using a pre-trained model state.')
parser.add_argument('--ckpt_path', type=str, metavar='N', default='', \
help='Path to ckpt with saved model state to be loaded. Only effective if --from_ckpt == True.')

args = parser.parse_args()

torch.manual_seed(1)
img_range = [0,100]

#set up saving directory
if args.save_dir =='':
    args.save_dir = os.getcwd()
if args.save_dir != '' and not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
else:
    pass"""

torch.manual_seed(1)
img_range = [0,100]

if __name__ == '__main__':
    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    """torch.cuda.set_per_process_memory_fraction(0.5, 0)
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    # less than 0.5 will be ok:
    tmp_tensor = torch.empty(int(total_memory * 0.499), dtype=torch.int8, device='cuda')
    del tmp_tensor
    torch.cuda.empty_cache()
    # this allocation will raise a OOM:
    #torch.empty(total_memory // 2, dtype=torch.int8, device='cuda')

    print(f"Using device: {device}")

    # Additional Info when using cuda
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(torch.cuda.get_device_name(0))
        torch.cuda.set_per_process_memory_fraction(1.0)
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    print(torch.cuda.mem_get_info(device=None))"""

    epochs = 1
    from_ckpt = False
    save_freq = 1
    test_freq = 1
    batch_size = 6
    data_dir = "./Data/Videos/"
    save_dir = "./Saves/22vid/"

    main_start = time.time()
    loaders_dict = data.setup_data_loaders(batch_size=batch_size,data_dir=data_dir)
    model = vae.BehaviourVAE(device_name=device,save_dir=save_dir)
    if from_ckpt == True:
        print(os.getcwd())
        assert os.path.exists(from_ckpt), 'Oops, looks like ckpt file given does NOT exist!'
        print('='*40)
        print('Loading model state from: {}'.format(from_ckpt))
        model.load_state(filename = from_ckpt)
    model.train_loop(loaders_dict, epochs=epochs, save_freq=save_freq,test_freq=test_freq)
    projections = model.get_latent_umap(loaders_dict, save_dir="", title="Latent Space plot")
    recons = model.get_recons(loaders_dict['dset'], [img_range[0], img_range[1]])
    tooltip.tooltip_plot_DC(loaders_dict['dset'], projections, recons, output_dir = save_dir, type='orig', n=len(loaders_dict['train'].dataset), img_range=img_range)
    main_end = time.time()
    print('Total model runtime (seconds): {}'.format(main_end - main_start))
