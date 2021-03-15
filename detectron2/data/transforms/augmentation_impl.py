# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Implement many useful :class:`Augmentation`.
"""
import numpy as np
import sys
import random
import copy
import math
from PIL import Image
from detectron2.data.detection_utils import adjust_scale, rand_by_rate, get_inv_tuple_matrix, get_random_color, get_paste_point
from .augmentation import Augmentation, _transform_to_aug
from .transform import *

__all__ = [
    "RandomApply",
    "RandomBrightness",
    "RandomContrast",
    "RandomCrop",
    "RandomExtent",
    "RandomFlip",
    "RandomSaturation",
    "RandomLighting",
    "RandomRotation",
    "Resize",
    "ResizeShortestEdge",
    "RandomCrop_CategoryAreaConstraint",
    "EdgeFilter",
    "ResizeRatio",
    "BoxShear",
    "BoxContrast",
    "BoxErase",
    "Noise",
    "Mosaic",
    "BoxAttentionCrop",
    "BoxMove",
]


class RandomApply(Augmentation):
    """
    Randomly apply an augmentation with a given probability.
    """

    def __init__(self, tfm_or_aug, prob=0.5):
        """
        Args:
            tfm_or_aug (Transform, Augmentation): the transform or augmentation
                to be applied. It can either be a `Transform` or `Augmentation`
                instance.
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
        """
        super().__init__()
        self.aug = _transform_to_aug(tfm_or_aug)
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"
        self.prob = prob

    def get_transform(self, *args):
        do = self._rand_range() < self.prob
        if do:
            return self.aug.get_transform(*args)
        else:
            return NoOpTransform()

    def __call__(self, aug_input):
        do = self._rand_range() < self.prob
        if do:
            return self.aug(aug_input)
        else:
            return NoOpTransform()


class RandomFlip(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


class Resize(Augmentation):
    """ Resize image to a fixed target size"""

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: PIL interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, image):
        return ResizeTransform(
            image.shape[0], image.shape[1], self.shape[0], self.shape[1], self.interp
        )


class ResizeShortestEdge(Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return ResizeTransform(h, w, newh, neww, self.interp)


class RandomRotation(Augmentation):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around the given center.
    """

    def __init__(self, prob, angle, expand=True, center=None, sample_style="range", interp=None):
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)
        self._init(locals())

    def get_transform(self, image):
        do = self._rand_range() < self.prob
        if do:
            h, w = image.shape[:2]
            center = None
            if self.is_range:
                angle = np.random.uniform(self.angle[0], self.angle[1])
                if self.center is not None:
                    center = (
                        np.random.uniform(self.center[0][0], self.center[1][0]),
                        np.random.uniform(self.center[0][1], self.center[1][1]),
                    )
            else:
                angle = np.random.choice(self.angle)
                if self.center is not None:
                    center = np.random.choice(self.center)

            if center is not None:
                center = (w * center[0], h * center[1])  # Convert to absolute coordinates

            if angle % 360 == 0:
                return NoOpTransform()

            return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)
        else:
            return NoOpTransform()


class RandomCrop(Augmentation):
    """
    Randomly crop a subimage out of an image.
    """

    def __init__(self, crop_type: str, crop_size):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
                See `config/defaults.py` for explanation.
            crop_size (tuple[float]): the relative ratio or absolute pixels of
                height and width
        """
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return CropTransform(w0, h0, cropw, croph)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))


class RandomCrop_CategoryAreaConstraint(Augmentation):
    """
    Similar to :class:`RandomCrop`, but find a cropping window such that no single category
    occupies a ratio of more than `single_category_max_area` in semantic segmentation ground
    truth, which can cause unstability in training. The function attempts to find such a valid
    cropping window for at most 10 times.
    """

    def __init__(
        self,
        crop_type: str,
        crop_size,
        single_category_max_area: float = 1.0,
        ignored_category: int = None,
    ):
        """
        Args:
            crop_type, crop_size: same as in :class:`RandomCrop`
            single_category_max_area: the maximum allowed area ratio of a
                category. Set to 1.0 to disable
            ignored_category: allow this category in the semantic segmentation
                ground truth to exceed the area ratio. Usually set to the category
                that's ignored in training.
        """
        self.crop_aug = RandomCrop(crop_type, crop_size)
        self._init(locals())

    def get_transform(self, image, sem_seg):
        if self.single_category_max_area >= 1.0:
            return self.crop_aug.get_transform(image)
        else:
            h, w = sem_seg.shape
            for _ in range(10):
                crop_size = self.crop_aug.get_crop_size((h, w))
                y0 = np.random.randint(h - crop_size[0] + 1)
                x0 = np.random.randint(w - crop_size[1] + 1)
                sem_seg_temp = sem_seg[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
                labels, cnt = np.unique(sem_seg_temp, return_counts=True)
                if self.ignored_category is not None:
                    cnt = cnt[labels != self.ignored_category]
                if len(cnt) > 1 and np.max(cnt) < np.sum(cnt) * self.single_category_max_area:
                    break
            crop_tfm = CropTransform(x0, y0, crop_size[1], crop_size[0])
            return crop_tfm


class RandomExtent(Augmentation):
    """
    Outputs an image by cropping a random "subrect" of the source image.

    The subrect can be parameterized to include pixels outside the source image,
    in which case they will be set to zeros (i.e. black). The size of the output
    image will vary with the size of the random subrect.
    """

    def __init__(self, scale_range, shift_range):
        """
        Args:
            output_size (h, w): Dimensions of output image
            scale_range (l, h): Range of input-to-output size scaling factor
            shift_range (x, y): Range of shifts of the cropped subrect. The rect
                is shifted by [w / 2 * Uniform(-x, x), h / 2 * Uniform(-y, y)],
                where (w, h) is the (width, height) of the input image. Set each
                component to zero to crop at the image's center.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        img_h, img_w = image.shape[:2]

        # Initialize src_rect to fit the input image.
        src_rect = np.array([-0.5 * img_w, -0.5 * img_h, 0.5 * img_w, 0.5 * img_h])

        # Apply a random scaling to the src_rect.
        src_rect *= np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Apply a random shift to the coordinates origin.
        src_rect[0::2] += self.shift_range[0] * img_w * (np.random.rand() - 0.5)
        src_rect[1::2] += self.shift_range[1] * img_h * (np.random.rand() - 0.5)

        # Map src_rect coordinates into image coordinates (center at corner).
        src_rect[0::2] += 0.5 * img_w
        src_rect[1::2] += 0.5 * img_h

        return ExtentTransform(
            src_rect=(src_rect[0], src_rect[1], src_rect[2], src_rect[3]),
            output_size=(int(src_rect[3] - src_rect[1]), int(src_rect[2] - src_rect[0])),
        )


class RandomContrast(Augmentation):
    """
    Randomly transforms image contrast.

    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, prob, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        do = self._rand_range() < self.prob
        if do:
            w = np.random.uniform(self.intensity_min, self.intensity_max)
            return BlendTransform(src_image=image.mean(), src_weight=1 - w, dst_weight=w)
        else:
            return NoOpTransform()


class RandomBrightness(Augmentation):
    """
    Randomly transforms image brightness.

    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, prob, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        do = self._rand_range() < self.prob
        if do:
            w = np.random.uniform(self.intensity_min, self.intensity_max)
            return BlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)
        else:
            return NoOpTransform()


class RandomSaturation(Augmentation):
    """
    Randomly transforms saturation of an RGB image.
    Input images are assumed to have 'RGB' channel order.

    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation (make the image more grayscale)
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation (1 preserves input).
            intensity_max (float): Maximum augmentation (1 preserves input).
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        assert image.shape[-1] == 3, "RandomSaturation only works on RGB images"
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        grayscale = image.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)


class RandomLighting(Augmentation):
    """
    The "lighting" augmentation described in AlexNet, using fixed PCA over ImageNet.
    Input images are assumed to have 'RGB' channel order.

    The degree of color jittering is randomly sampled via a normal distribution,
    with standard deviation given by the scale parameter.
    """

    def __init__(self, scale):
        """
        Args:
            scale (float): Standard deviation of principal component weighting.
        """
        super().__init__()
        self._init(locals())
        self.eigen_vecs = np.array(
            [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])

    def get_transform(self, image):
        assert image.shape[-1] == 3, "RandomLighting only works on RGB images"
        weights = np.random.normal(scale=self.scale, size=3)
        return BlendTransform(
            src_image=self.eigen_vecs.dot(weights * self.eigen_vals), src_weight=1.0, dst_weight=1.0
        )


class EdgeFilter(Augmentation):
    def __init__(self, prob, min_radius, max_radius):
        super().__init__()
        self._init(locals())
    
    def get_transform(self, image):
        do = self._rand_range() < self.prob
        if do:
            radius = random.uniform(self.min_radius, self.max_radius)
            return EdgeFilterTransform(radius=radius)
        else:
            return NoOpTransform()


class ResizeRatio(Augmentation):
    """Generate a randn aspect_ratio between [min, point] or [point, max], then resize the img 
    """
    def __init__(self, prob, min_ratio, max_ratio, balanced_point, balanced_transform, keep_same_size):
        """Args:
        prob: (float) do or not prob
        min_ratio: (float) min aspect ratio
        max_ratio: (float) max aspect ratio
        balanced_point: (float) the point between min and max
        balanced_transform: (bool) if true, generate ar between [min,point] or [point,max] with prob 0.5; if false, between [min, max]
        keep_same_size: (bool) 
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        do = rand_by_rate(self.prob)
        if do:
            h, w = image.shape[:2]
            dst_w, dst_h = adjust_scale((w,h), self.min_ratio, self.max_ratio, self.balanced_point, self.balanced_transform, self.keep_same_size)
            return ResizeTransform(h, w, dst_h, dst_w)
        else:
            return NoOpTransform()


class BoxShear(Augmentation):
    """Shear some boxes along x or y axis
    """
    def __init__(self, shear_prob_x, shear_prob_y, shear_min_len_x, shear_max_len_x, shear_min_len_y, shear_max_len_y, balanced_transform, balanced_point, min_area_rate):
        """Args:
        shear_prob_x: (list) rate to shear along x axis(for class0 - classN)
        shear_prob_y: (list) rate to shear along y axis(for class0 - classN)
        shear_min_len_x: (int) the min value for shear along x axis
        shear_max_len_x: (int) the max value for shear along x axis
        shear_min_len_y: (int) the min value for shear along y axis
        shear_max_len_y: (int) the max value for shear along y axis
        balanced_transform: (bool) if true, generate shear value between [min,point] or [point,max] with prob 0.5; if false, between [min, max]
        balanced_point: (int) the point between min and max
        min_area_rate:(float):
        """
        super().__init__()
        self._init(locals())
    
    def get_transform(self, image, annotations):
        along_x_info = []
        along_y_info = []
        #loop each gt box, then decide whether do shear by its category and prob 
        for anno_idx, anno in enumerate(annotations):
            do_shear_x = rand_by_rate(self.shear_prob_x[anno["category_id"]])
            do_shear_y = rand_by_rate(self.shear_prob_y[anno["category_id"]])
            box_w = anno["bbox"][2] - anno["bbox"][0]
            box_h = anno["bbox"][3] - anno["bbox"][1]
            if do_shear_x:
                #compute shear params
                # compute shear length along x axis
                real_shear_min_len_x = self.shear_min_len_x
                real_shear_max_len_x = self.shear_max_len_x
                if self.shear_min_len_x<self.balanced_point and self.shear_max_len_x>self.balanced_point and self.balanced_transform:
                    if rand_by_rate(0.5):
                        real_shear_min_len_x = self.balanced_point
                    else:
                        real_shear_max_len_x = self.balanced_point
                shear_len_x = random.uniform(real_shear_min_len_x, real_shear_max_len_x)
                x_axis_shift = abs(shear_len_x) * box_h
                new_w = box_w + int(round(x_axis_shift))
                # compute rotation matrix
                # why M looks like this? transform is: new_x=x+shear_len_x*y, new_y=y. each pixel will move a distance(x_axis_shift).
                # if shear_len_x is negtive, pixel will be out of new img(new_w, h), we have to do a translation(x_axis_shift) to move it back. 
                M = np.array([[1, shear_len_x, x_axis_shift if shear_len_x < 0 else 0], [0, 1, 0]])
                M = get_inv_tuple_matrix(M)
                color = get_random_color()

                along_x_info.append({"anno_idx":anno_idx, "M":M, "new_w":new_w, "color":color})
            if do_shear_y:
                #compute shear params
                real_shear_min_len_y = self.shear_min_len_y
                real_shear_max_len_y = self.shear_max_len_y
                if self.shear_min_len_y<self.balanced_point and self.shear_max_len_y>self.balanced_point and self.balanced_transform:
                    if rand_by_rate(0.5):
                        real_shear_min_len_y = self.balanced_point
                    else:
                        real_shear_max_len_y = self.balanced_point
                shear_len_y = random.uniform(real_shear_min_len_y, real_shear_max_len_y)
                y_axis_shift = abs(shear_len_y) * box_w
                new_h = box_h + int(round(y_axis_shift))
                M = np.array([[1, 0, 0], [shear_len_y, 1, y_axis_shift if shear_len_y < 0 else 0]])
                M = get_inv_tuple_matrix(M)
                color = get_random_color()

                along_y_info.append({"anno_idx":anno_idx, "M":M, "new_h":new_h, "color":color})

        if len(along_x_info)==0 and len(along_y_info)==0:
            return NoOpTransform()
        else:
            return BoxShearTransform(along_x_info, along_y_info, annotations)


class BoxContrast(Augmentation):
    """Do contrast to some boxes
    """
    def __init__(self, prob, side_rate_w, side_rate_h, left_side_upper_bound, right_side_lower_bound):
        """
        prob: (list) rate to do contrast(for class0 - classN)
        side_rate_w: (float) decide whether pick a random region along width
        side_rate_h: (float) decide whether pick a random region along height
        left_side_upper_bound: (float) the max value for random pick, for left side
        right_side_lower_bound: (float) the min value for random pick, for right side
        """
        super().__init__()
        self._init(locals())
    
    def get_transform(self, image, annotations):
        invert_region = []
        for anno_idx, anno in enumerate(annotations):
            do = rand_by_rate(self.prob[anno["category_id"]])
            if do:
                side_w_do = rand_by_rate(self.side_rate_w)
                side_h_do = rand_by_rate(self.side_rate_h)
                bbox = copy.deepcopy(anno["bbox"])
                box_w = bbox[2] - bbox[0]
                box_h = bbox[3] - bbox[1]
                #pick a region along width
                if side_w_do:
                    bbox[0] = bbox[0] + box_w*random.uniform(0, self.left_side_upper_bound)
                    bbox[2] = bbox[2] - box_w*(1-random.uniform(self.right_side_lower_bound, 1))
                #pick a region along height
                if side_h_do:
                    bbox[1] = bbox[1] + box_h*random.uniform(0, self.left_side_upper_bound)
                    bbox[3] = bbox[3] - box_h*(1-random.uniform(self.right_side_lower_bound, 1))
                
                bbox = list(map(int, bbox))
                invert_region.append(bbox)

        if len(invert_region) == 0:
            return NoOpTransform()
        else:
            return BoxContrastTransform(invert_region)


class BoxErase(Augmentation):
    """Fill a random part of a box.
    """
    def __init__(self, prob):
        """Args:
        prob: (float) rate to remove a part of box
        """
        super().__init__()
        self._init(locals())
    
    def get_transform(self, image, annotations):
        fill_region = []
        for anno in annotations:
            do = rand_by_rate(self.prob)
            if do:
                bbox = anno["bbox"]
                #assert(bbox[0]<bbox[2] and bbox[1]<bbox[3])
                #compute a region to fill
                box_w = bbox[2] - bbox[0]
                box_h = bbox[3] - bbox[1]
                if box_w<20 or box_h<20:
                    continue

                region_box = []
                region_w = random.randint(1, box_w//2)
                region_h = random.randint(1, box_h//2)
                region_left = random.randint(1, box_w//2)
                region_top = random.randint(1, box_h//2)

                region_box.append(region_left + bbox[0])
                region_box.append(region_top + bbox[1])
                region_box.append(region_left + bbox[0] + region_w)
                region_box.append(region_top + bbox[1] + region_h)
                
                region_box = list(map(int, region_box))
                fill_region.append(region_box)
        
        if len(fill_region) == 0:
            return NoOpTransform()
        else:
            return BoxEraseTransform(fill_region)


class Noise(Augmentation):
    """Add some noise on img
    """
    def __init__(self, prob, min_rate, max_rate):
        """Args:
        prob(float): prob to add noise
        min_rate(float): use to generate how much noise to add
        max_rate(float): use to generate how much noise to add
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        do = rand_by_rate(self.prob)
        if do:
            rate = random.uniform(self.min_rate, self.max_rate)

            return NoiseTransform(rate)
        else:
            return NoOpTransform()


class Mosaic(Augmentation):
    """Tile img like a mosaic. max_move is adviced to <= 2.  If there is an img A, this func may yield thus result:
    AA    A     AA
          A     AA
    """
    def __init__(self, prob, max_move):
        """Args:
        prob(float): rate to do mosaic
        max_move(int): max times to do mosaic
        """
        super().__init__()
        self._init(locals())
    
    def get_transform(self, image, annotations):
        mosaic_direction = []
        for i in range(self.max_move):
            do = rand_by_rate(self.prob)
            if not do:
                continue
            #mosaic to right or bottom
            mosaic_right = rand_by_rate(0.5)
            if mosaic_right:
                mosaic_direction.append("right")
            else:
                mosaic_direction.append("bottom")

        if len(mosaic_direction) == 0:
            return NoOpTransform()
        else:
            return MosaicTransform(mosaic_direction, annotations)


class BoxAttentionCrop(Augmentation):
    """Normal order is 'resize img' -> 'crop img', but raw img maybe too big, resize will cost too much time,
    thus crop first then resize.
    """
    def __init__(self, prob, crop_size, balance_by_cls, largest_max_absolute_area, min_scale, max_scale, balanced_transform, balanced_point, box_resize_info, min_area, max_area, min_area_rate):
        """
        prob(float): probility to do this aug
        crop_size(list): [h, w], just use to compute a temp crop size, the followed Resize aug will really change img to this size
        balance_by_cls(bool): if true, each category has the same prob to be picked; if false, each box has the same prob to be picked
        largest_max_absolute_area(int): the area of resized box can not beyond this
        min_scale(float): if dont do attention crop, use this to change img size
        max_scale(float): if dont do attention crop, use this to change img size
        balanced_transform(bool): 
        balanced_point(float):
        min_area_rate(float): after crop, box erea < this rate will be removed
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image, annotations):
        #remove some boxes
        #img, boxes, labels = utils.remove_region(img, boxes, labels, min_area, max_area) 
        old_img_h, old_img_w = image.shape[:2]
        crop_h, crop_w = self.crop_size
        new_img_w = 0
        new_img_h = 0
        new_crop_w = 0
        new_crop_h = 0
        crop_left = 0
        crop_top = 0
        random_idx = -1 #which GT box to do attention
        random_class = -1 #category of this box
        categories = [anno["category_id"] for anno in annotations]
        do = rand_by_rate(self.prob) and len(annotations)>0

        #compute crop width and height
        if do:
            if self.balance_by_cls:
                #compute how many defferent classes are contained by this img
                contain_classes = []
                for label in categories:
                    if label not in contain_classes:
                        contain_classes.append(label)
                num_classes = len(contain_classes)
                assert(num_classes > 0)
                #pick a class with same probility
                random_idx = random.randint(0, num_classes-1)
                random_class = contain_classes[random_idx]
                #pick a GT which is this class
                contain_classes = []
                for i, label in enumerate(categories):
                    if label == random_class:
                        contain_classes.append(i)
                random_idx = contain_classes[random.randint(0, len(contain_classes)-1)]
            else:
                random_idx = random.randint(0, len(categories)-1)
                random_class = categories[random_idx]
            #select the box_resize_info param according to picked class
            info_idx = 0
            for i in range(len(self.box_resize_info)):
                if self.box_resize_info[i]['metric_class_id'] == random_class:
                    info_idx = i
                    break
            #compute crop params according to this box_resize_info
            max_rescale = self.box_resize_info[info_idx]['max_rescale']
            target_min_absolute_area = self.box_resize_info[info_idx]['target_min_absolute_area']
            target_max_absolute_area = self.box_resize_info[info_idx]['target_max_absolute_area']
            scale_idx = random.randint(0, len(target_min_absolute_area)-1)
            #random pick an area region
            target_area = random.uniform(target_min_absolute_area[scale_idx], target_max_absolute_area[scale_idx])
            raw_area = (annotations[random_idx]["bbox"][2]-annotations[random_idx]["bbox"][0]) * (annotations[random_idx]["bbox"][3]-annotations[random_idx]["bbox"][1])
            #final rescale ratio
            calced_rescale_ratio = target_area / raw_area
            final_resize_ratio = min(max_rescale, calced_rescale_ratio)
            #compute new size
            new_crop_w = min(int(crop_w / math.pow(final_resize_ratio, 0.5)), old_img_w)
            new_crop_h = min(int(crop_h / math.pow(final_resize_ratio, 0.5)), old_img_h)
        else:
            max_scale_rate = -1.
            assert(self.largest_max_absolute_area>0)
            #compute the max box area in this img
            max_box_area = -1.
            for anno in annotations:
                max_box_area = max(max_box_area, (anno["bbox"][2]-anno["bbox"][0])*(anno["bbox"][3]-anno["bbox"][1]))
            
            if max_box_area > 0:
                max_scale_rate = pow(self.largest_max_absolute_area/max_box_area, 0.5)
            else:
                return NoOpTransform()
            
            #compute real scale size
            real_min_scale = self.min_scale
            real_max_scale = self.max_scale
            real_min_scale = min(real_min_scale, max_scale_rate)
            real_max_scale = min(real_max_scale, max_scale_rate)

            if real_min_scale<self.balanced_point and real_max_scale>self.balanced_point and self.balanced_transform:
                if rand_by_rate(0.5):
                    real_min_scale = self.balanced_point
                else:
                    real_max_scale = self.balanced_point
            final_resize_ratio = random.uniform(real_min_scale, real_max_scale)
            #compute new size
            new_crop_w = min(int(crop_w / math.pow(final_resize_ratio, 0.5)), old_img_w)
            new_crop_h = min(int(crop_h / math.pow(final_resize_ratio, 0.5)), old_img_h)
        
        #compute crop left and top
        if old_img_w==new_crop_w and old_img_h==new_crop_h:
            return NoOpTransform()
        if do:
            x_min, y_min, x_max, y_max = annotations[random_idx]["bbox"]
            #left
            if new_crop_w <= x_max-x_min+1:
                crop_left = x_min
            else:
                random_left = max(0,int(x_max-new_crop_w))
                random_right = min(int(x_min), int(old_img_w-new_crop_w))
                if random_left >= random_right:
                    return NoOpTransform()
                crop_left = random.randint(random_left, random_right)
            #top
            if new_crop_h <= y_max-y_min+1:
                crop_top = y_min
            else:
                tmp_top1 = max(0,int(y_max-new_crop_h))
                tmp_top2 = min(int(y_min), int(old_img_h-new_crop_h))
                if tmp_top1>=tmp_top2:
                    crop_top = tmp_top2
                else:
                    crop_top = random.randint(tmp_top1, tmp_top2) 
        else:
            crop_left = random.randint(0, old_img_w - new_crop_w + 1)
            crop_top  = random.randint(0, old_img_h - new_crop_h + 1)
        
        #check
        if crop_left+new_crop_w > old_img_w:
            crop_left = old_img_w - new_crop_w
        if crop_top+new_crop_h > old_img_h:
            crop_top = old_img_h - new_crop_h

        return CropTransform(int(crop_left), int(crop_top), int(new_crop_w), int(new_crop_h), self.min_area_rate)


class BoxMove(Augmentation):
    """Copy some boxes and paste them to other region on img
    """
    def __init__(self, prob, remove_rate, max_move, min_move_rescale, max_move_rescale, balanced_point, balanced_transform, side_rate):
        """Args:
        prob(list): prob to move(for class0 - classN)
        remove_rate(float): rate to remove this box
        max_move(int): max times to move one box
        min_move_rescale(float): after move, rescale one box, refer to min_ratio in atom_resize_ratio
        max_move_rescale(float): after move, rescale one box, refer to max_ratio in atom_resize_ratio
        balanced_point(float): the point between min and max
        balanced_transform(bool): if true, generate ar between [min,point] or [point,max] with prob 0.5; if false, between [min, max]
        side_rate(float): rate to move one box to its side
        """
        super().__init__()
        self._init(locals())
    
    def get_transform(self, image, annotations):
        img_h, img_w = image.shape[:2]
        move_info = [] # anno_id:[]
        moved_once = False
        for idx, anno in enumerate(annotations):
            anno_trans_info = {}
            anno_trans_info["anno_id"] = idx
            # some classes dont need to move
            if self.prob[anno["category_id"]] == 0:
                continue

            just_remove_it = rand_by_rate(self.remove_rate)

            anno_move_info = []
            for _ in range(self.max_move):
                do = rand_by_rate(self.prob[anno["category_id"]])
                if not do:
                    continue
                
                moved_once = True
                single_move_info = {}
                single_move_info["raw_box_coor"] = anno["bbox"]
                # set rescale ratio
                box_w = int(anno["bbox"][2] - anno["bbox"][0])
                box_h = int(anno["bbox"][3] - anno["bbox"][1])
                dst_size = adjust_scale((box_w, box_h), self.min_move_rescale, self.max_move_rescale, self.balanced_point, self.balanced_transform, True)
                dst_size[0] = min(dst_size[0], img_w)
                dst_size[1] = min(dst_size[1], img_h)
                single_move_info["dst_size"] = dst_size
                # start move process
                mv_side = rand_by_rate(self.side_rate)
                #move to side
                to_paste_point = [-1, -1] # left top
                if mv_side:
                    if rand_by_rate(0.5): #top
                        to_paste_point[0] = max(0, anno["bbox"][0]-(dst_size[0]-box_w)/2)
                        to_paste_point[1] = max(0, anno["bbox"][1]-dst_size[1])
                    else: #bottom
                        to_paste_point[0] = max(0, anno["bbox"][0]-(dst_size[0]-box_w)/2)
                        to_paste_point[1] = min(anno["bbox"][3], img_h-dst_size[1])
                #move to anywhere
                else:
                    to_paste_point = get_paste_point((img_w, img_h), dst_size)
                
                single_move_info["to_paste_point"] = to_paste_point

                anno_move_info.append(single_move_info)
            
            anno_trans_info["anno_move_info"] = anno_move_info
        
            move_info.append(anno_trans_info)
        
        if not moved_once:
            return NoOpTransform()
        else:
            return BoxMoveTransform(move_info)