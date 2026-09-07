# Conference Grouped Split Report

## Protocol

- One canonical VSN image retained per SHA-256 content hash.
- Perceptual neighbours with pHash Hamming distance <= 4 assigned to one group.
- Each group assigned wholly to train, validation, or test.
- ASN audio follows the group of its paired image content.

## Results

- Input VSN rows: 44094
- Exact duplicate rows removed: 5692
- Canonical VSN images: 38402
- Perceptual groups: 33371
- Largest group: 11 images
- Groups larger than one: 4729
- Label-conflicting near links not merged: 0
- Missing ASN pairs: 0
- Cross-split group errors: 0
- VSN split counts: {'val': 5761, 'train': 26880, 'test': 5761}
- ASN split counts: {'test': 596, 'train': 2868, 'val': 630}

## Required Next Step

Retrain VSN and ASN local encoders from scratch on these indices before repeating shared-layer experiments. Existing checkpoints have seen the old random splits and cannot be used for the final publication result.