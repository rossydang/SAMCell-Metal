import numpy as np
import json
import os
import argparse

def main(path: str, name: str, version: str) -> None:
    imgs = np.load('imgs.npy')
    anns = np.load('anns.npy')

    os.makedirs(f'{path}_v{version}', exist_ok=True)
    images = []
    for i in range(len(imgs)):
        img = imgs[i]
        ann = anns[i]

        np.save(os.path.join(f'{path}_v{version}', f'img_{i}.npy'), img)
        np.save(os.path.join(f'{path}_v{version}', f'ann_{i}.npy'), ann)

        filename = os.path.join(f'{path}_v{version}', f'img_{i}.npy')
        width = img.shape[1]
        height = img.shape[0]

        annotations = [
            {
                'name': 'multi_mask',
                'mask': {
                    'path': os.path.join(f'{path}_v{version}', f'ann_{i}.npy'),
                    'index' : {f'{i}': f'cell_{i}' for i in np.unique(ann) if i != 0} # exclude background
                }
            }
        ]

        images.append(
            {
                'filename': filename,
                'width': width,
                'height': height,
                'annotations': annotations
            }
        )

    darwin_dict = {
        'dataset': {'name': name, 'version': version},
        'images': images
    }

    with open(f'{name}_v{version}.json', 'w') as f:
        json.dump(darwin_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''A script to generate a Darwin V7 compatible JSON object for mask segmentation dataset.
        All images should be stored in the working directory as `imgs.npy` as a (N, W, H) array, and the corresponding multi-mask annotations as 'anns.py'.
        Creates a JSON file and a file directory that must accompany the JSON when used with Darwin API.
        '''
    )

    parser.add_argument(
        '-p', '--path',
        type=str,
        default='DATASET_Darwin',
        help="The path (relative to the working directory) to store the new data to be packaged with the Darwin JSON. The dataset will be stored as PATH_vVERSION. (Default: 'DATASET_Darwin')"
    )

    parser.add_argument(
        '-d', '--dataset',
        type=str,
        default='DATASET',
        help="The dataset name. This name will be included in the Darwin V7 metadata. (Default: 'DATASET')"
    )

    parser.add_argument(
        '-v', '--version',
        type=str,
        default='1.0',
        help="The version of the dataset. Version will be included in the Darwin V7 metadata. (Default: '1.0')"
    )

    args = parser.parse_args()

    main(path=args.path, name=args.dataset, version=args.version)