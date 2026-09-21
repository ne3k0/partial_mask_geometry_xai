import numpy as np
import os

OLD_BASE = '/gpfs/scratch/qp251874/workspaces/icassp_2027/layerwise_attribution_data'
NEW_BASE = os.path.join(os.path.expanduser('~'), 'partial_mask_geometry_xai', 'data', 'test_data')
clip = 'YzZU-PJA5Woo'
cls = 'bagpipes'
fills = ['zero', 'mean', 'gaussian_noise']

for model in ['panns_specaug_trained', 'panns_no_specaug', 'ast_wrapper']:
    print(f'\n=== {model} ===')
    old = os.path.join(OLD_BASE, model, cls, 'embs')
    new = os.path.join(NEW_BASE, model, cls, 'embs')

    old_orig = np.load(os.path.join(old, f'{clip}_original_logits.npy'))
    new_orig = np.load(os.path.join(new, f'{clip}_original_sigmoid.npy'))
    print(f'  original sigmoid : allclose={np.allclose(old_orig, new_orig, atol=1e-6)}')

    for fill in fills:
        old_npz = np.load(os.path.join(old, f'{clip}_perturbations_{fill}.npz'))
        new_npz = np.load(os.path.join(new, f'{clip}_perturbations_{fill}.npz'))
        new_sig = np.load(os.path.join(new, f'{clip}_perturb_{fill}.npy'))
        old_foc = np.load(os.path.join(old, f'{clip}_fully_occluded_{fill}_logits.npy'))
        new_foc = np.load(os.path.join(new, f'{clip}_foc_{fill}.npy'))

        occ = np.array_equal(old_npz['occ_fracs'], new_npz['occ_fracs'])
        sig = np.allclose(old_npz['logits'], new_sig, atol=1e-6)
        foc = np.allclose(old_foc, new_foc, atol=1e-6)

        print(f'  [{fill:14s}] occ_fracs={occ}  sigmoid={sig}  foc={foc}')
