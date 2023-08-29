# brain-segmantation
ct segmented images of blocked brain nurves 

from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/Brain_Segment
!ls

pip install simpleITK

import os
import SimpleITK as sitk

data_dir = "/content/drive/MyDrive/Brain_Segment"  # Update this with your data directory

# List all folders (each containing a pair of CT image and segmentation mask)
slice_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
slice_folders

# for debugging
# os.path.join(data_dir, slice_folder, slice_folders[0]+ ".nii.gz")

# for folder in slice_folders:
#     slice_path = os.path.join(data_dir, folder)
#     ct_image_path = os.path.join(slice_path, ".nii.gz") #"ct_image.nii.gz")
#     segmentation_path = os.path.join(slice_path, "_ROI.nii.gz") #"segmentation_mask.nii.gz")
#     print(slice_path)
#     print(ct_image_path)
#     print(segmentation_path)

for files in slice_folders:
    slice_path = os.path.join(data_dir, files)
    ct_image_path = os.path.join(slice_path, files + ".nii.gz") #"ct_image.nii.gz")
    segmentation_path = os.path.join(slice_path, files + "_ROI.nii.gz") #"segmentation_mask.nii.gz")

    # Load CT image and segmentation mask
    ct_image = sitk.ReadImage(ct_image_path)
    segmentation_mask = sitk.ReadImage(segmentation_path)

    # You can perform further processing or segmentation here
    # For example, applying preprocessing or feeding the images to a segmentation model

    # Printing some information about the loaded images
    print(f"Slice Folder: {files}")
    print(f"CT Image Size: {ct_image.GetSize()}")
    print(f"Segmentation Mask Size: {segmentation_mask.GetSize()}")
    print("=" * 30)

    # for debug
# ct_image_path

import matplotlib.pyplot as plt
import numpy as np
def show_slice_window(slice, level, window):
   """
   Function to display an image slice
   Input is a numpy 2D array
   """
   max = level + window/2
   min = level - window/2
   slice = slice.clip(min,max)
   plt.figure()
   plt.imshow(slice.T, cmap="gray", origin="lower")
   plt.savefig('L'+str(level)+'W'+str(window))

import nibabel as nib
ct_img = nib.load("/content/drive/MyDrive/Brain_Segment/Anon2/Anon2.nii.gz")
print(ct_img.header)

def find_pix_dim(ct_img):
    """
    Get the pixdim of the CT image.
    A general solution that gets the pixdim indicated from the image dimensions. From the last 2 image dimensions, we get their pixel dimension.
    Args:
        ct_img: nib image

    Returns: List of the 2 pixel dimensions
    """
    pix_dim = ct_img.header["pixdim"] # example [1,2,1.5,1,1]
    dim = ct_img.header["dim"] # example [1,512,512,1,1]
    max_indx = np.argmax(dim)
    pixdimX = pix_dim[max_indx]
    dim = np.delete(dim, max_indx)
    pix_dim = np.delete(pix_dim, max_indx)
    max_indy = np.argmax(dim)
    pixdimY = pix_dim[max_indy]
    return [pixdimX, pixdimY] # example [2, 1.5]

    def intensity_seg(ct_numpy, min=-1000, max=-300):
   clipped = clip_ct(ct_numpy, min, max)
   return measure.find_contours(clipped, 0.95)

 
   
   def find_brains(contours):
    """
    Chooses the contours that correspond to the brain
    First, we exclude non-closed sets-contours
    Then we assume some min area and volume to exclude small contours
    Then the body is excluded as the highest volume closed set
    The remaining areas correspond to the brain
    Args:
        contours: all the detected contours

    Returns: contours that correspond to the brain area
    """
    brain_contours = []
    vol_contours = []

    for contour in contours:
        hull = ConvexHull(contour)

       # set some constraints for the volume
        if hull.volume > 2000 and set_is_closed(contour):
            brain_contours.append(contour)
            vol_contours.append(hull.volume)


    if len(brain_contours) == 2:
        return brain_contours_contours
    elif len(brain_contours) > 2:
        vol_contours, brain_contours = (list(t) for t in
                zip(*sorted(zip(vol_contours, brain_contours))))
        brain_contours.pop(-1)
    return brain_contours

    import numpy as np
from PIL import Image, ImageDraw
def create_mask_from_polygon(image, contours):
    """
    Creates a binary mask with the dimensions of the image and
    converts the list of polygon-contours to binary masks and merges them together
    Args:
        image: the image that the contours refer to
        contours: list of contours

    Returns:

    """
    brain_mask = np.array(Image.new('L', image.shape, 0))
    for contour in contours:
        x = contour[:, 0]
        y = contour[:, 1]
        polygon_tuple = list(zip(x, y))
        img = Image.new('L', image.shape, 0)
        ImageDraw.Draw(img).polygon(polygon_tuple, outline=0, fill=1)
        mask = np.array(img)
        brain_mask += mask

    brain_mask[brain_mask > 1] = 1  # sanity check to make 100% sure that the mask is binary
    return brain_mask.T  # transpose it to be aligned with the image dims



    def save_nifty(img_np, name, affine):
    """
    binary masks should be converted to 255 so it can be displayed in a nii viewer
    we pass the affine of the initial image to make sure it exits in the same
    image coordinate space
    Args:
        img_np: the binary mask
        name: output name
        affine: 4x4 np array
    Returns:
    """
    img_np[img_np == 1] = 255
    ni_img = nib.Nifti1Image(img_np, affine)
    nib.save(ni_img, name + '.nii.gz')


    def create_vessel_mask(brain_mask, ct_numpy, denoise=False):
    vessels = brain_mask * ct_numpy  # isolate brain area
    vessels[vessels == 0] = -1000
    vessels[vessels >= -500] = 1
    vessels[vessels < -500] = 0
    show_slice(vessels)
    if denoise:
        return denoise_vessels(lungs_contour, vessels)
    show_slice(vessels)
    return vessels


    def overlay_plot(im, mask):
    plt.figure()
    plt.imshow(im.T, 'gray', interpolation='none')
    plt.imshow(mask.T, 'jet', interpolation='none', alpha=0.5)




def compute_area(mask, pixdim):
    """
    Computes the area (number of pixels) of a binary mask and multiplies the pixels
    with the pixel dimension of the acquired CT image
    Args:
        brain_mask: binary brain mask
        pixdim: list or tuple with two values

    Returns: the brain area in mm^2
    """
    mask[mask >= 1] = 1
    brain_pixels = np.sum(mask)
    return brain_pixels * pixdim[0] * pixdim[1]






    train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)


output="/content/drive/MyDrive/Brain_Segment"
