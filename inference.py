import argparse
import collections
import glob
import shutil
import sys
from datetime import datetime
from pathlib import Path
import PIL.Image as Image
import torch
from torchvision.transforms import transforms
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('infernece')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        patch_size=config['data_loader']['args']['patch_size'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            batch_size = data_loader.batch_size
            indices = torch.arange(batch_size * i, batch_size * i + data.shape[0])
            pred = torch.argmax(output, dim=1)
            data_loader.dataset.patches.store_data(indices, [pred.unsqueeze(1)])

    preds = [(data_loader.dataset.patches.fuse_data(idx, 0).cpu(), data_loader.dataset.data[idx])
             for idx in range(len(data_loader.dataset.data))]
    trsfm = transforms.ToPILImage()

    out_dir = list(config.save_dir.parts)
    out_dir[-3] = 'output'
    out_dir = Path(*out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pred, path in preds:
        filename = Path(path).stem + '.png'
        pred = trsfm(pred.float())
        pred.save(out_dir / filename)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--data', default=None, type=str,
                      help='path to data (default: None)')

    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    dst_data = Path('.data/', run_id)

    data_dir = dst_data / 'test' / 'images'
    masks_dir = dst_data / 'test' / 'masks'

    data_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    if '--data' in sys.argv:
        src_data = Path(sys.argv[sys.argv.index('--data') + 1])
        if src_data.is_file():
            shutil.copy(src_data, data_dir)
            mask = Image.new('1', Image.open(src_data).size)
            mask.save(masks_dir / (src_data.stem + '.png'), 'PNG')
        else:
            for filename in glob.glob('*.jpg', root_dir=src_data):
                file_path = src_data / filename
                if file_path.is_file():
                    shutil.copy(file_path, data_dir)
                    mask = Image.new('1', Image.open(file_path).size)
                    mask.save(masks_dir / (file_path.stem + '.png'), 'PNG')
        sys.argv += ['--data_dir', str(dst_data)]

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--data_dir'], type=str, target='data_loader;args;data_dir')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
    shutil.rmtree(dst_data)
