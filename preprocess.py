# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 13:22:16 2021

@author: Hanan
"""
# Imports
from __future__ import division
import numpy as np
import h5py, sys
from collections import namedtuple
import skimage.morphology as morph
import skimage.transform
import skimage.draw
import skimage.morphology as morph
from skimage.feature import canny
import matplotlib.pyplot as plt
import cv2
from cv2 import bilateralFilter
from skimage.restoration import denoise_tv_chambolle
from skimage.restoration import denoise_tv_bregman
import SimpleITK as sitk
import scipy.ndimage as snd
import numpy as np
import h5py
import os
from collections import namedtuple
rng = np.random.RandomState(40)

def roi_patch_transform_norm(data, transformation, nlabel, random_augmentation_params=None,
                             mm_center_location=(.5, .4), mm_patch_size=(128, 128), mask_roi=False,
                             uniform_scale=False, random_denoise=False, denoise=False, ACDC=True):

    # Input data dimension is of shape: (X,Y)
    add_noise = transformation.get('add_noise', None)
    patch_size = transformation['patch_size']
    mm_patch_size = transformation.get('mm_patch_size', mm_patch_size)
    mask_roi = transformation.get('mask_roi', mask_roi)


    image = data['image'][:]
    label = data['label'][:] 
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)
    # pixel spacing in X and Y
    pixel_spacing = data['pixel_spacing'][:]
    roi_center = data['roi_center'][:]
    roi_radii = data['roi_radii'][:]

    # Check if the roi fits patch_size else resize the image to patch dimension
    max_radius = roi_radii[1]
    if not CheckImageFitsInPatch(image, roi_center, max_radius, patch_size):
        mm_patch_size = (256, 256)


    # if random_augmentation_params=None -> sample new params
    # if the transformation implies no augmentations then random_augmentation_params remains None
    if not random_augmentation_params:
        random_augmentation_params = sample_augmentation_parameters(transformation)
        # print random_augmentation_params
    # build scaling transformation
    current_shape = image.shape[:2]

    # scale ROI radii and find ROI center in normalized patch
    if roi_center.any():
        mm_center_location = tuple(int(r * ps) for r, ps in zip(roi_center, pixel_spacing))

    # scale the images such that they all have the same scale if uniform_scale=True
    norm_rescaling = 1./ pixel_spacing[0] if uniform_scale else 1
    mm_shape = tuple(int(float(d) * ps) for d, ps in zip(current_shape, pixel_spacing))

    tform_normscale = build_rescale_transform(downscale_factor=norm_rescaling,
                                              image_shape=current_shape, target_shape=mm_shape)

    tform_shift_center, tform_shift_uncenter = build_shift_center_transform(image_shape=mm_shape,
                                                                            center_location=mm_center_location,
                                                                            patch_size=mm_patch_size)
    patch_scale = max(1. * mm_patch_size[0] / patch_size[0],
                      1. * mm_patch_size[1] / patch_size[1])
    tform_patch_scale = build_rescale_transform(patch_scale, mm_patch_size, target_shape=patch_size)

    total_tform = tform_patch_scale + tform_shift_uncenter + tform_shift_center + tform_normscale

    # build random augmentation
    if random_augmentation_params:
        augment_tform = build_augmentation_transform(rotation=random_augmentation_params.rotation,
                                                     shear=random_augmentation_params.shear,
                                                     translation=random_augmentation_params.translation,
                                                     flip_x=random_augmentation_params.flip_x,
                                                     flip_y=random_augmentation_params.flip_y,
                                                     zoom=random_augmentation_params.zoom)
        total_tform = tform_patch_scale + tform_shift_uncenter + augment_tform + tform_shift_center + tform_normscale
        # print total_tform.params
    if add_noise is not None:
        noise_type = add_noise[rng.randint(len(add_noise))]
        image = generate_noisy_image(noise_type, image)  
    # For Multi-Channel Data warp all the slices in the same manner
    n_channels = image.shape[2]
    transformed_image = np.zeros(patch_size+(n_channels,))
    for i in range(n_channels):   
        transformed_image[:,:,i] = fast_warp(normalize(image[:,:,i]), total_tform, output_shape=patch_size, mode='symmetric')
    image = transformed_image
    label = multilabel_transform(label, total_tform, patch_size, nlabel)


    if denoise:
        if random_denoise:
            image = rng.randint(2) > 0 if denoise else image
        else:  
            image = denoise_img_vol(image)

    # apply transformation to ROI and mask the images
    if roi_center.any() and roi_radii.any() and mask_roi:
        roi_scale = random_augmentation_params.roi_scale if random_augmentation_params else 1.6  # augmentation
        roi_zoom = random_augmentation_params.zoom if random_augmentation_params else pixel_spacing
        rescaled_roi_radii = (roi_scale * roi_radii[0], roi_scale * roi_radii[1])
        out_roi_radii = (int(roi_zoom[0] * rescaled_roi_radii[0] * pixel_spacing[0] / patch_scale),
                         int(roi_zoom[1] * rescaled_roi_radii[1] * pixel_spacing[1] / patch_scale))
        roi_mask = make_circular_roi_mask(patch_size, (patch_size[0] / 2, patch_size[1] / 2), out_roi_radii)
        image *= roi_mask

    if random_augmentation_params:
        if uniform_scale:
            targets_zoom_factor = random_augmentation_params.zoom[0] * random_augmentation_params.zoom[1]
        else:
            targets_zoom_factor = pixel_spacing[0]*pixel_spacing[1]*\
                                random_augmentation_params.zoom[0]*random_augmentation_params.zoom[1]
    else:
        targets_zoom_factor = pixel_spacing[0]*pixel_spacing[1]
    return image, label, targets_zoom_factor

def build_rescale_transform(downscale_factor, image_shape, target_shape):
    """
    estimating the correct rescaling transform is slow, so just use the
    downscale_factor to define a transform directly. This probably isn't
    100% correct, but it shouldn't matter much in practice.
    """
    rows, cols = image_shape
    trows, tcols = target_shape
    tform_ds = skimage.transform.AffineTransform(scale=(downscale_factor, downscale_factor))
    # centering
    shift_x = cols / (2.0 * downscale_factor) - tcols / 2.0
    shift_y = rows / (2.0 * downscale_factor) - trows / 2.0
    tform_shift_ds = skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))
    return tform_shift_ds + tform_ds

def CheckImageFitsInPatch(image, roi, max_radius, patch_size):
    boFlag = True
    max_radius += max_radius*.05
    if (max_radius > patch_size[0]/2) or (max_radius > patch_size[1]/2)\
        or (image.shape[0]>=512) or (image.shape[1]>=512):
        # print('The patch wont fit the roi: resize image', image.shape, max_radius, patch_size[0]/2)
        boFlag = False
    return boFlag

def sample_augmentation_parameters(transformation):
    # This code does random sampling from the transformation parameters
    # Random number generator
    if set(transformation.keys()) == {'patch_size', 'mm_patch_size'} or \
                    set(transformation.keys()) == {'patch_size', 'mm_patch_size', 'mask_roi'}:
        return None

    shift_x = rng.uniform(*transformation.get('translation_range_x', [0., 0.]))
    shift_y = rng.uniform(*transformation.get('translation_range_y', [0., 0.]))
    translation = (shift_x, shift_y)
    rotation = rng.uniform(*transformation.get('rotation_range', [0., 0.]))
    shear = rng.uniform(*transformation.get('shear_range', [0., 0.]))
    roi_scale = rng.uniform(*transformation.get('roi_scale_range', [1., 1.]))
    z = rng.uniform(*transformation.get('zoom_range', [1., 1.]))
    zoom = (z, z)

    if 'do_flip' in transformation:
        if type(transformation['do_flip']) == tuple:
            flip_x = rng.randint(2) > 0 if transformation['do_flip'][0] else False
            flip_y = rng.randint(2) > 0 if transformation['do_flip'][1] else False
        else:
            flip_x = rng.randint(2) > 0 if transformation['do_flip'] else False
            flip_y = False
    else:
        flip_x, flip_y = False, False

    sequence_shift = rng.randint(30) if transformation.get('sequence_shift', False) else 0

    return namedtuple('Params', ['translation', 'rotation', 'shear', 'zoom',
                                 'roi_scale',
                                 'flip_x', 'flip_y',
                                 'sequence_shift'])(translation, rotation, shear, zoom,
                                                    roi_scale,
                                                    flip_x, flip_y,
                                                    sequence_shift)


                                                    
def PreProcessData(file_name, data, mode, transformation_params, Alternate=True):
    """
    Preprocess the image, ground truth (label) and return  along with its corresponding weight map
    """
    image = data['image'][:]
    label = data['label'][:] 
    roi = data['roi_center'][:]
    roi_radii = data['roi_radii'][:]
    pixel_spacing = data['pixel_spacing'][:]
    n_labels = transformation_params['n_labels']
    max_radius = roi_radii[1]
    patch_size = transformation_params[mode]['patch_size']
    max_size = transformation_params.get('data_crop_pad', (256, 256))

    # print (image.shape, pixel_spacing)

    if transformation_params['full_image']:
        # Dont do any ROI crop or augmentation
        # Just make sure that all the images are of fixed size
        # By cropping or Padding
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        patch_image = resize_image_with_crop_or_pad_3D(normalize(image), max_size[0], max_size[1])
        patch_label = resize_image_with_crop_or_pad_3D(label[:,:,None], max_size[0], max_size[1])
    else: 
        # If to alternate randomly between training with and without augmentation
        if Alternate:
            boAlternate = rng.randint(2) > 0
        else:
            boAlternate = False

        if not transformation_params['data_augmentation'] or boAlternate:
            # Check if the roi fits patch_size else resize the image to patch dimension
            if CheckImageFitsInPatch(image, roi, max_radius, patch_size):
                # Center around roi
                patch_image = crop_img_patch_from_roi(normalize(image), roi, patch_size)
                patch_label = crop_img_patch_from_roi(label, roi, patch_size)
                # If patch size does not fit then pad or crop
                patch_image = resize_image_with_crop_or_pad_3D(patch_image[:,:, None], patch_size[0], patch_size[1])
                patch_label = resize_image_with_crop_or_pad_3D(patch_label[:,:, None], patch_size[0], patch_size[1])
                # print (patch_image.shape, patch_label.shape)
            else:
                patch_image  = crop_img_patch_from_roi(normalize(image), roi, max_size)
                patch_label = crop_img_patch_from_roi(label, roi, max_size)
                patch_image = resize_sitk_2D(patch_image, patch_size)
                patch_label = resize_sitk_2D(patch_label, patch_size, interpolator=sitk.sitkNearestNeighbor)
        else:
            random_params = sample_augmentation_parameters(transformation_params[mode])
            # print (random_params)
            patch_image, patch_label, _ = roi_patch_transform_norm(data, transformation_params[mode], 
                                        n_labels, random_augmentation_params=random_params,
                                        uniform_scale=False, random_denoise=False, denoise=False)

            if transformation_params['data_deformation'] and (rng.randint(2) > 0)\
            and (transformation_params[mode] != 'valid'):
                patch_image, patch_label = produceRandomlyDeformedImage(patch_image, patch_label[:,:,None])                       

    # Expand dimensions to feed to network
    if patch_image.ndim == 2:
        patch_image = np.expand_dims(patch_image, axis=2) 
    if patch_label.ndim == 3:
        patch_label = np.squeeze(patch_label, axis=2) 
    patch_image = np.expand_dims(patch_image, axis=0)
    patch_label = np.expand_dims(patch_label, axis=0)
    # print (patch_image.shape, patch_label.shape)
    # TODO: Check post nrmalization effects
    # patch_image = normalize(patch_image, scheme='zscore')
    weight_map = getEdgeEnhancedWeightMap(patch_label, label_ids=range(n_labels), scale=1, edgescale=1,  assign_equal_wt=False)
    return (patch_image, patch_label, weight_map)

def ExtractProcessedData(path , mode , transformation_params):
        data = h5py.File(path,'r')
        file_name = os.path.basename(path)
        # print (file_name)
        # Preprocessing of Input Image and Label
        patch_img, patch_gt, patch_wmap = PreProcessData(file_name, data, mode, transformation_params)
        return patch_img, patch_gt, patch_wmap, file_name
    
def build_shift_center_transform(image_shape, center_location, patch_size):
    """Shifts the center of the image to a given location.
    This function tries to include as much as possible of the image in the patch
    centered around the new center. If the patch around the ideal center
    location doesn't fit within the image, we shift the center to the right so
    that it does.
    params in (i,j) coordinates !!!
    """
    if center_location[0] < 1. and center_location[1] < 1.:
        center_absolute_location = [
            center_location[0] * image_shape[0], center_location[1] * image_shape[1]]
    else:
        center_absolute_location = [center_location[0], center_location[1]]

    # Check for overlap at the edges
    center_absolute_location[0] = max(
        center_absolute_location[0], patch_size[0] / 2.0)
    center_absolute_location[1] = max(
        center_absolute_location[1], patch_size[1] / 2.0)

    center_absolute_location[0] = min(
        center_absolute_location[0], image_shape[0] - patch_size[0] / 2.0)

    center_absolute_location[1] = min(
        center_absolute_location[1], image_shape[1] - patch_size[1] / 2.0)

    # Check for overlap at both edges
    if patch_size[0] > image_shape[0]:
        center_absolute_location[0] = image_shape[0] / 2.0
    if patch_size[1] > image_shape[1]:
        center_absolute_location[1] = image_shape[1] / 2.0

    # Build transform
    new_center = np.array(center_absolute_location)
    translation_center = new_center - 0.5
    translation_uncenter = -np.array((patch_size[0] / 2.0, patch_size[1] / 2.0)) - 0.5
    return (
        skimage.transform.SimilarityTransform(translation=translation_center[::-1]),
        skimage.transform.SimilarityTransform(translation=translation_uncenter[::-1]))

def build_augmentation_transform(rotation=0, shear=0, translation=(0, 0), flip_x=False, flip_y=False, zoom=(1.0, 1.0)):
    if flip_x:
        shear += 180  # shear by 180 degrees is equivalent to flip along the X-axis
    if flip_y:
        shear += 180
        rotation += 180

    tform_augment = skimage.transform.AffineTransform(scale=(1. / zoom[0], 1. / zoom[1]), rotation=np.deg2rad(rotation),
                                                      shear=np.deg2rad(shear), translation=translation)
    return tform_augment

def generate_noisy_image(noise_typ, image):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
        noise = ['gauss', 'poisson', 's&p', 'speckle', 'denoise', 'none1', 'none2']            
    """
    if noise_typ == "gauss":
        row,col = image.shape[:2]
        mean = 0
        var = 0.0001
        sigma = var**0.5
        gauss = rng.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col,1)
        noisy = image + gauss
        return noisy

    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [rng.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [rng.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = rng.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ =="speckle":
        row,col = image.shape[:2]
        gauss = 0.1*rng.randn(row,col)
        gauss = gauss.reshape(row,col,1)        
        noisy = image + image * gauss
        return noisy   
    else:
        return image  

def fast_warp(img, tf, output_shape, mode='constant', order=1):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params  # tf._matrix is
    # TODO: check if required
    # mode='symmetric'
    return skimage.transform.warp(img, m, output_shape=output_shape, mode=mode, order=order)

def normalize(image, scheme='zscore'):
    # Do Image Normalization:
    if scheme == 'zscore':
        image = normalize_zscore(image, z=0.5, offset=0)
    elif scheme == 'minmax':
        image = normalize_minmax(image)
    elif scheme == 'truncated_zscore':
        image = normalize_zscore(image, z=2, offset=0.5, clip=True)
    else:
        image = image
    return image

def normalize_zscore(data, z=2, offset=0.5, clip=False):
    """
    Normalize contrast across volume
    """
    mean = np.mean(data)
    std = np.std(data)
    img = ((data - mean) / (2 * std * z) + offset) 
    if clip:
        # print ('Before')
        # print (np.min(img), np.max(img))
        img = np.clip(img, -0.0, 1.0)
        # print ('After clip')
        # print (np.min(img), np.max(img))
    return img

def normalize_minmax(data):
    """
    Normalize contrast across volume
    """
    _min = np.float(np.min(data))
    _max = np.float(np.max(data))
    if (_max-_min)!=0:
        img = (data - _min) / (_max-_min)
    else:
        img = np.zeros_like(data)            
    return img

def slicewise_normalization(img_data4D, scheme='minmax'):
    """
    Do slice-wise normalization for the 4D image data(3D+ Time)
    """
    x_dim, y_dim, n_slices, n_phases = img_data4D.shape

    data_4d = np.zeros([x_dim, y_dim, n_slices, n_phases])
    for slice in range(n_slices):
        for phase in range(n_phases):
            data_4d[:,:,slice, phase] = normalize(img_data4D[:,:,slice, phase], scheme)
    return data_4d

def phasewise_normalization(img_data4D, scheme='minmax'):
    """
    Do slice-wise normalization for the 4D image data(3D+ Time)
    """
    x_dim, y_dim, n_slices, n_phases = img_data4D.shape

    data_4d = np.zeros([x_dim, y_dim, n_slices, n_phases])
    for phase in range(n_phases):
        data_4d[:,:,:, phase] = normalize(img_data4D[:,:,:, phase], scheme)
    return data_4d

def multilabel_transform(img, tf, output_shape, nlabel, mode='constant', order=0):
    """
    Binarize images do apply transform on each of the binary images and take argmax while
    doing merge operation
    Order -> 0 : nearest neighbour interpolation
    """
    bin_img_stack = multilabel_binarize(img, nlabel)
    n_labels = len(bin_img_stack)
    tf_bin_img_stack = np.zeros((n_labels,) + output_shape, dtype='uint8')
    for label in range(n_labels):
        tf_bin_img_stack[label] = fast_warp(bin_img_stack[label], tf, output_shape=output_shape, mode=mode, order=order)
    # Do merge operation along the axis = 0
    return np.argmax(tf_bin_img_stack, axis=0)

def multilabel_binarize(image_2D, nlabel):
    """
    Binarize multilabel images and return stack of binary images
    Returns: Tensor of shape: Bin_Channels* Image_shape(3D tensor)
    TODO: Labels are assumed to discreet in steps from -> 0,1,2,...,nlabel-1
    """
    labels = range(nlabel)
    out_shape = (len(labels),) + image_2D.shape
    bin_img_stack = np.ones(out_shape, dtype='uint8')
    for label in labels:
        bin_img_stack[label] = np.where(image_2D == label, bin_img_stack[label], 0)
    return bin_img_stack

def produceRandomlyDeformedImage(image, label, numcontrolpoints=2, stdDef=15):
    sitkImage=sitk.GetImageFromArray(image, isVector=False)
    sitklabel=sitk.GetImageFromArray(label, isVector=False)

    transfromDomainMeshSize=[numcontrolpoints]*sitkImage.GetDimension()

    tx = sitk.BSplineTransformInitializer(sitkImage,transfromDomainMeshSize)


    params = tx.GetParameters()

    paramsNp=np.asarray(params,dtype=float)
    paramsNp = paramsNp + np.random.randn(paramsNp.shape[0])*stdDef
    #remove z deformations! The resolution in z is too bad in case of 3D or its channels in 2D
    paramsNp[0:int(len(params)/3)]=0 #remove z deformations! The resolution in z is too bad in case of 3D or its channels

    params=tuple(paramsNp)
    tx.SetParameters(params)
    # print (sitkImage.GetSize(), sitklabel.GetSize(), transfromDomainMeshSize, paramsNp.shape)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)

    outimgsitk = resampler.Execute(sitkImage)

    # For Label use nearest neighbour
    resampler.SetReferenceImage(sitklabel)
    resampler.SetInterpolator(sitk.sitkLabelGaussian)
    resampler.SetDefaultPixelValue(0)
    outlabsitk = resampler.Execute(sitklabel)

    outimg = sitk.GetArrayFromImage(outimgsitk)
    outimg = outimg.astype(dtype=np.float32)

    outlbl = sitk.GetArrayFromImage(outlabsitk)
    outlbl = (outlbl).astype(dtype=np.uint8)
    return outimg, outlbl

selem = morph.disk(1)
def getEdgeEnhancedWeightMap(label, label_ids =[0,1,2,3], scale=1, edgescale=1, assign_equal_wt=False):
    shape = (0,)+label.shape[1:]
    weight_map = np.empty(shape, dtype='uint8')
    if assign_equal_wt:
        return np.ones_like(label)
    for i in range(label.shape[0]): 
        #Estimate weight maps:
        weights = np.ones(len(label_ids))
        slice_map = np.ones(label[i,:,:].shape)
        for _id in label_ids:
            class_frequency = np.sum(label[i,:,:] == label_ids[_id])
            if class_frequency:
                weights[label_ids.index(_id)] = scale*label[i,:,:].size/class_frequency
                slice_map[np.where(label[i,:,:]==label_ids.index(_id))] = weights[label_ids.index(_id)]
                edge = np.float32(morph.binary_dilation(
                    canny(np.float32(label[i,:,:]==label_ids.index(_id)),sigma=1), selem=selem))
                edge_frequency = np.sum(np.sum(edge==1.0))
                if edge_frequency:    
                    slice_map[np.where(edge==1.0)] += edgescale*label[i,:,:].size/edge_frequency
            # print (weights)
            # utils.imshow(edge, cmap='gray')
        # utils.imshow(weight_map, cmap='gray')
        weight_map = np.append(weight_map, np.expand_dims(slice_map, axis=0), axis=0)
    return np.float32(weight_map)
