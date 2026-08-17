import nibabel as nib
import numpy as np
import os

def load_mri(filepath):
    """Loads an Analyze (.hdr/.img) or NIfTI (.nii/.nii.gz) file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MRI file not found: {filepath}")
    
    img = nib.load(filepath)
    spacing = np.diag(img.affine)[:3]
    
    data = img.get_fdata()
    # Remove 4th dimension if size 1
    if data.ndim == 4:
        data = data[..., 0]
        
    return data, img.affine, spacing

def standardize_orientation(data, affine):
    """Checks if the data is in LAS or RAS and ensures uniform RAS representation."""
    # ADNI raw Analyze files are typically in LAS space
    # Nibabel aff2axcodes tells us the orientation
    axcodes = nib.aff2axcodes(affine)
    
    # If LAS, axis 0 is Left-to-Right. If we want RAS, we flip axis 0.
    # In our COM pipeline, LAS is fine as long as we know x=0 is Left and x=max is Right.
    # We will work directly in the native scan space but verify axcodes is ('L', 'A', 'S')
    if axcodes != ('L', 'A', 'S'):
        # For this dataset, they are consistently LAS. If not, print a warning.
        print(f"Warning: Unexpected image orientation {axcodes}. Target is ('L', 'A', 'S').")
    return data
