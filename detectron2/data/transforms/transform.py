# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# File: transform.py
import inspect
import pprint
from typing import Callable, List, TypeVar

from fvcore.transforms.transform import Transform
from fvcore.transforms.transform_util import to_float_tensor, to_numpy

import numpy as np
import random
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter
from detectron2.data.detection_utils import fill_region, compute_crop_box_iou
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

__all__ = [
    "ExtentTransform",
    "ResizeTransform",
    "RotationTransform",
    "ColorTransform",
    "PILColorTransform",
    "CropTransform",
    "BlendTransform",
    "VFlipTransform",
    "HFlipTransform",
    "NoOpTransform",
    "TransformList",
    "ScaleTransform",
    "GridSampleTransform",
    "EdgeFilterTransform",
    "BoxShearTransform",
    "BoxContrastTransform",
    "NoiseTransform",
    "BoxEraseTransform",
    "MosaicTransform",
    "BoxMoveTransform",
]


class WrapTransform(Transform):
    """Re-implement the Transform class, because it is a python pack, I can't add new func to it.
    Just add some custom func to Transform, and all the child transform classes will be a subclass of this
    """
    def apply_annotations(self, annotations):
        """
        For most transform methods, num of annos won't be changed, only the box coor will change.
        Thus, if num of annos changes, re-write this func.
        """
        boxes = np.array([anno["bbox"] for anno in annotations])
        boxes = self.apply_box(boxes).tolist()

        for i,box in enumerate(boxes):
            annotations[i]["bbox"] = box

        annotations = [annotations[i] for i,anno in enumerate(annotations) if (anno["bbox"][0]<anno["bbox"][2] and anno["bbox"][1]<anno["bbox"][3])]
        return annotations


_T = TypeVar("_T")


# pyre-ignore-all-errors
class TransformList(WrapTransform):
    """
    Maintain a list of transform operations which will be applied in sequence.
    Attributes:
        transforms (list[Transform])
    """

    def __init__(self, transforms: List[Transform]):
        """
        Args:
            transforms (list[Transform]): list of transforms to perform.
        """
        super().__init__()
        # "Flatten" the list so that TransformList do not recursively contain TransfomList.
        # The additional hierarchy does not change semantic of the class, but cause extra
        # complexities in e.g, telling whether a TransformList contains certain Transform
        tfms_flatten = []
        for t in transforms:
            assert isinstance(
                t, Transform
            ), f"TransformList requires a list of Transform. Got type {type(t)}!"
            if isinstance(t, TransformList):
                tfms_flatten.extend(t.transforms)
            else:
                tfms_flatten.append(t)
        self.transforms = tfms_flatten

    def _apply(self, x: _T, meth: str) -> _T:
        """
        Apply the transforms on the input.
        Args:
            x: input to apply the transform operations.
            meth (str): meth.
        Returns:
            x: after apply the transformation.
        """
        for t in self.transforms:
            x = getattr(t, meth)(x)
        return x

    def __getattribute__(self, name: str):
        # use __getattribute__ to win priority over any registered dtypes
        if name.startswith("apply_"):
            return lambda x: self._apply(x, name)
        return super().__getattribute__(name)

    def __add__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(self.transforms + others)

    def __iadd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        self.transforms.extend(others)
        return self

    def __radd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(others + self.transforms)

    def __len__(self) -> int:
        """
        Returns:
            Number of transforms contained in the TransformList.
        """
        return len(self.transforms)

    def __getitem__(self, idx) -> Transform:
        return self.transforms[idx]

    def inverse(self) -> "TransformList":
        """
        Invert each transform in reversed order.
        """
        return TransformList([x.inverse() for x in self.transforms[::-1]])

    def __repr__(self) -> str:
        msgs = [str(t) for t in self.transforms]
        return "TransformList[{}]".format(", ".join(msgs))

    __str__ = __repr__

    # The actual implementations are provided in __getattribute__.
    # But abstract methods need to be declared here.
    def apply_coords(self, x):
        raise NotImplementedError

    def apply_image(self, x):
        raise NotImplementedError


class HFlipTransform(WrapTransform):
    """
    Perform horizontal flip.
    """

    def __init__(self, width: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        # NOTE: opencv would be faster:
        # https://github.com/pytorch/pytorch/issues/16424#issuecomment-580695672
        if img.ndim <= 3:  # HxW, HxWxC
            return np.flip(img, axis=1)
        else:
            return np.flip(img, axis=-2)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        coords[:, 0] = self.width - coords[:, 0]
        return coords

    def inverse(self) -> Transform:
        """
        The inverse is to flip again
        """
        return self


class VFlipTransform(WrapTransform):
    """
    Perform vertical flip.
    """

    def __init__(self, height: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        tensor = torch.from_numpy(np.ascontiguousarray(img))
        if len(tensor.shape) == 2:
            # For dimension of HxW.
            tensor = tensor.flip((-2))
        elif len(tensor.shape) > 2:
            # For dimension of HxWxC, NxHxWxC.
            tensor = tensor.flip((-3))
        return tensor.numpy()

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        coords[:, 1] = self.height - coords[:, 1]
        return coords

    def inverse(self) -> Transform:
        """
        The inverse is to flip again
        """
        return self


class NoOpTransform(WrapTransform):
    """
    A transform that does nothing.
    """

    def __init__(self):
        super().__init__()

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def inverse(self) -> Transform:
        return self

    def __getattr__(self, name: str):
        if name.startswith("apply_"):
            return lambda x: x
        raise AttributeError("NoOpTransform object has no attribute {}".format(name))


class ScaleTransform(WrapTransform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h: int, w: int, new_h: int, new_w: int, interp: str = None):
        """
        Args:
            h, w (int): original image size.
            new_h, new_w (int): new image size.
            interp (str): interpolation methods. Options includes `nearest`, `linear`
                (3D-only), `bilinear`, `bicubic` (4D-only), and `area`.
                Details can be found in:
                https://pytorch.org/docs/stable/nn.functional.html
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Resize the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): interpolation methods. Options includes `nearest`, `linear`
                (3D-only), `bilinear`, `bicubic` (4D-only), and `area`.
                Details can be found in:
                https://pytorch.org/docs/stable/nn.functional.html

        Returns:
            ndarray: resized image(s).
        """
        if len(img.shape) == 4:
            h, w = img.shape[1:3]
        elif len(img.shape) in (2, 3):
            h, w = img.shape[:2]
        else:
            raise ("Unsupported input with shape of {}".format(img.shape))
        assert (
            self.h == h and self.w == w
        ), "Input size mismatch h w {}:{} -> {}:{}".format(self.h, self.w, h, w)
        interp_method = interp if interp is not None else self.interp
        # Option of align_corners is only supported for linear, bilinear,
        # and bicubic.
        if interp_method in ["linear", "bilinear", "bicubic"]:
            align_corners = False
        else:
            align_corners = None

        # note: this is quite slow for int8 images because torch does not
        # support it https://github.com/pytorch/pytorch/issues/5580
        float_tensor = torch.nn.functional.interpolate(
            to_float_tensor(img),
            size=(self.new_h, self.new_w),
            mode=interp_method,
            align_corners=align_corners,
        )
        return to_numpy(float_tensor, img.shape, img.dtype)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute the coordinates after resize.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: resized coordinates.
        """
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply resize on the full-image segmentation.

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: resized segmentation.
        """
        segmentation = self.apply_image(segmentation, interp="nearest")
        return segmentation

    def inverse(self) -> Transform:
        """
        The inverse is to resize it back.
        """
        return ScaleTransform(self.new_h, self.new_w, self.h, self.w, self.interp)


class GridSampleTransform(WrapTransform):
    def __init__(self, grid: np.ndarray, interp: str):
        """
        Args:
            grid (ndarray): grid has x and y input pixel locations which are
                used to compute output. Grid has values in the range of [-1, 1],
                which is normalized by the input height and width. The dimension
                is `N x H x W x 2`.
            interp (str): interpolation methods. Options include `nearest` and
                `bilinear`.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Apply grid sampling on the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): interpolation methods. Options include `nearest` and
                `bilinear`.
        Returns:
            ndarray: grid sampled image(s).
        """
        interp_method = interp if interp is not None else self.interp
        float_tensor = torch.nn.functional.grid_sample(
            to_float_tensor(img),  # NxHxWxC -> NxCxHxW.
            torch.from_numpy(self.grid),
            mode=interp_method,
            padding_mode="border",
            align_corners=False,
        )
        return to_numpy(float_tensor, img.shape, img.dtype)

    def apply_coords(self, coords: np.ndarray):
        """
        Not supported.
        """
        raise NotImplementedError()

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply grid sampling on the full-image segmentation.

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: grid sampled segmentation.
        """
        segmentation = self.apply_image(segmentation, interp="nearest")
        return segmentation


class CropTransform(WrapTransform):
    def __init__(self, x0: int, y0: int, w: int, h: int, min_area_rate: float):
        # TODO: flip the order of w and h.
        """
        Args:
            x0, y0, w, h (int): crop the image(s) by img[y0:y0+h, x0:x0+w].
            min_area_rate: a box after crop, if area/raw_aera<min_area_rate, remove this area.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: cropped image(s).
        """
        if len(img.shape) <= 3:
            return img[self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w]
        else:
            return img[..., self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w, :]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply crop transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: cropped coordinates.
        """
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords

    def apply_annotations(self, annotations):
        #raw box area
        boxes = np.array([anno["bbox"] for anno in annotations])
        raw_area = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
        if boxes.ndim == 1:
            boxes = boxes.reshape(-1, 4)
        #compute iou mask, and adjust labels
        crop_box = np.array([self.x0, self.y0, self.x0+self.w, self.y0+self.h])
        iou = compute_crop_box_iou(boxes, crop_box)
        mask = iou > 0
        boxes = boxes[mask]
        raw_area = raw_area[mask]

        #remove some annos
        annotations = [annotations[i] for i in range(len(annotations)) if mask[i]]

        #for iou>0, compute inter boxes
        crop_box = np.tile(crop_box, (boxes.shape[0], 1))
        inter_boxes = np.zeros_like(boxes)

        inter_boxes[:,0] = np.maximum(boxes[:,0], crop_box[:,0])
        inter_boxes[:,1] = np.maximum(boxes[:,1], crop_box[:,1])
        inter_boxes[:,2] = np.minimum(boxes[:,2], crop_box[:,2])
        inter_boxes[:,3] = np.minimum(boxes[:,3], crop_box[:,3])

        #cvt inter boxes' coors to crop img
        inter_boxes[:,0] = inter_boxes[:,0] - crop_box[:,0]
        inter_boxes[:,1] = inter_boxes[:,1] - crop_box[:,1]
        inter_boxes[:,2] = inter_boxes[:,2] - crop_box[:,0]
        inter_boxes[:,3] = inter_boxes[:,3] - crop_box[:,1]

        #new box area
        new_area = (inter_boxes[:,2]-inter_boxes[:,0]) * (inter_boxes[:,3]-inter_boxes[:,1])
        mask = new_area/raw_area > self.min_area_rate
        inter_boxes = inter_boxes[mask]

        if len(inter_boxes) >3:
            if inter_boxes[0][0] == inter_boxes[1][0] and inter_boxes[0][1] == inter_boxes[1][1] and inter_boxes[0][2] == inter_boxes[1][2] and inter_boxes[0][3] == inter_boxes[1][3]:
                print(inter_boxes)
                print(boxes)
                print(crop_box)
                assert(False)
        #remove some annos
        annotations = [annotations[i] for i in range(len(annotations)) if mask[i]]

        #update coor
        for i in range(len(annotations)):
            annotations[i]["bbox"] = inter_boxes[i]
        
        return annotations

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply crop transform on a list of polygons, each represented by a Nx2 array.
        It will crop the polygon with the box, therefore the number of points in the
        polygon might change.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            ndarray: cropped polygons.
        """
        import shapely.geometry as geometry

        # Create a window that will be used to crop
        crop_box = geometry.box(
            self.x0, self.y0, self.x0 + self.w, self.y0 + self.h
        ).buffer(0.0)

        cropped_polygons = []

        for polygon in polygons:
            polygon = geometry.Polygon(polygon).buffer(0.0)
            # polygon must be valid to perform intersection.
            assert polygon.is_valid, polygon
            cropped = polygon.intersection(crop_box)
            if cropped.is_empty:
                continue
            if not isinstance(cropped, geometry.collection.BaseMultipartGeometry):
                cropped = [cropped]
            # one polygon may be cropped to multiple ones
            for poly in cropped:
                # It could produce lower dimensional objects like lines or
                # points, which we want to ignore
                if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                    continue
                coords = np.asarray(poly.exterior.coords)
                # NOTE This process will produce an extra identical vertex at
                # the end. So we remove it. This is tested by
                # `tests/test_data_transform.py`
                cropped_polygons.append(coords[:-1])
        return [self.apply_coords(p) for p in cropped_polygons]


class BlendTransform(WrapTransform):
    """
    Transforms pixel colors with PIL enhance functions.
    """

    def __init__(self, src_image: np.ndarray, src_weight: float, dst_weight: float):
        """
        Blends the input image (dst_image) with the src_image using formula:
        ``src_weight * src_image + dst_weight * dst_image``

        Args:
            src_image (ndarray): Input image is blended with this image
            src_weight (float): Blend weighting of src_image
            dst_weight (float): Blend weighting of dst_image
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Apply blend transform on the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): keep this option for consistency, perform blend would not
                require interpolation.
        Returns:
            ndarray: blended image(s).
        """
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = self.src_weight * self.src_image + self.dst_weight * img
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            return self.src_weight * self.src_image + self.dst_weight * img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation

    def inverse(self) -> Transform:
        """
        The inverse is a no-op.
        """
        return NoOpTransform()




class ExtentTransform(WrapTransform):
    """
    Extracts a subregion from the source image and scales it to the output size.

    The fill color is used to map pixels from the source rect that fall outside
    the source image.

    See: https://pillow.readthedocs.io/en/latest/PIL.html#PIL.ImageTransform.ExtentTransform
    """

    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
        """
        Args:
            src_rect (x0, y0, x1, y1): src coordinates
            output_size (h, w): dst image size
            interp: PIL interpolation methods
            fill: Fill color used when src_rect extends outside image
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        h, w = self.output_size
        ret = Image.fromarray(img).transform(
            size=(w, h),
            method=Image.EXTENT,
            data=self.src_rect,
            resample=interp if interp else self.interp,
            fill=self.fill,
        )
        return np.asarray(ret)

    def apply_coords(self, coords):
        # Transform image center from source coordinates into output coordinates
        # and then map the new origin to the corner of the output image.
        h, w = self.output_size
        x0, y0, x1, y1 = self.src_rect
        new_coords = coords.astype(np.float32)
        new_coords[:, 0] -= 0.5 * (x0 + x1)
        new_coords[:, 1] -= 0.5 * (y0 + y1)
        new_coords[:, 0] *= w / (x1 - x0)
        new_coords[:, 1] *= h / (y1 - y0)
        new_coords[:, 0] += 0.5 * w
        new_coords[:, 1] += 0.5 * h
        return new_coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class ResizeTransform(WrapTransform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        assert len(img.shape) <= 4

        if img.dtype == np.uint8:
            pil_image = Image.fromarray(img)
            interp_method = interp if interp is not None else self.interp
            pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
            ret = np.asarray(pil_image)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {Image.BILINEAR: "bilinear", Image.BICUBIC: "bicubic"}
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[self.interp]
            img = F.interpolate(img, (self.new_h, self.new_w), mode=mode, align_corners=False)
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)


class RotationTransform(WrapTransform):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around its center.
    """

    def __init__(self, h, w, angle, expand=True, center=None, interp=None):
        """
        Args:
            h, w (int): original image size
            angle (float): degrees for rotation
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
        """
        super().__init__()
        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center
        if interp is None:
            interp = cv2.INTER_LINEAR
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        if expand:
            # find the new width and height bounds
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self._set_attributes(locals())
        self.rm_coords = self.create_rotation_matrix()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0 or self.angle % 360 == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        interp = interp if interp is not None else self.interp
        return cv2.warpAffine(img, self.rm_image, (self.bound_w, self.bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        """
        coords = np.asarray(coords, dtype=float)
        if len(coords) == 0 or self.angle % 360 == 0:
            return coords
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation

    def create_rotation_matrix(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm

    def inverse(self):
        """
        The inverse is to rotate it back with expand, and crop to get the original shape.
        """
        if not self.expand:  # Not possible to inverse if a part of the image is lost
            raise NotImplementedError()
        rotation = RotationTransform(
            self.bound_h, self.bound_w, -self.angle, True, None, self.interp
        )
        crop = CropTransform(
            (rotation.bound_w - self.w) // 2, (rotation.bound_h - self.h) // 2, self.w, self.h
        )
        return TransformList([rotation, crop])


class ColorTransform(WrapTransform):
    """
    Generic wrapper for any photometric transforms.
    These transformations should only affect the color space and
        not the coordinate space of the image (e.g. annotation
        coordinates such as bounding boxes should not be changed)
    """

    def __init__(self, op):
        """
        Args:
            op (Callable): operation to be applied to the image,
                which takes in an ndarray and returns an ndarray.
        """
        if not callable(op):
            raise ValueError("op parameter should be callable")
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        return self.op(img)

    def apply_coords(self, coords):
        return coords

    def inverse(self):
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        return segmentation


class PILColorTransform(ColorTransform):
    """
    Generic wrapper for PIL Photometric image transforms,
        which affect the color space and not the coordinate
        space of the image
    """

    def __init__(self, op):
        """
        Args:
            op (Callable): operation to be applied to the image,
                which takes in a PIL Image and returns a transformed
                PIL Image.
                For reference on possible operations see:
                - https://pillow.readthedocs.io/en/stable/
        """
        if not callable(op):
            raise ValueError("op parameter should be callable")
        super().__init__(op)

    def apply_image(self, img):
        img = Image.fromarray(img)
        return np.asarray(super().apply_image(img))


class EdgeFilterTransform(WrapTransform):
    """tfy
    Filter an image by using Gaussian Filter
    """
    def __init__(self, radius):
        super().__init__()
        self._set_attributes(locals())
    
    def apply_image(self, img):
        ret = Image.fromarray(img)
        ret = ret.filter(ImageFilter.GaussianBlur(self.radius))
        return np.asarray(ret)
    
    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return NoOpTransform()


class BoxShearTransform(WrapTransform):
    """tfy
    Shear some boxes along x or y axis
    """
    def __init__(self, along_x_info, along_y_info, annotations):
        """Args:
        along_x_info(list[dict]): along x aixs, the info to do shear for each gt box
        along_x_info(list[dict]): along y aixs, the info to do shear for each gt box
        annotations(list[dict]): annotations info
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        ret = Image.fromarray(img)

        for x_info in self.along_x_info:
            M        = x_info["M"]
            new_w    = x_info["new_w"]
            color    = x_info["color"]
            anno_idx = x_info["anno_idx"]

            bbox = list(map(int, self.annotations[anno_idx]["bbox"]))
            box_w = bbox[2] - bbox[0]
            box_h = bbox[3] - bbox[1]
        
            #get box img to shear
            cropped_img = ret.crop(bbox)
            #do shear
            cropped_img = cropped_img.transform((int(new_w), int(box_h)), Image.AFFINE, M, resample=Image.BILINEAR, fillcolor=color)
            transformed_w, transformed_h = cropped_img.size
            if transformed_w<=0 or transformed_w<=0:
                return np.asarray(ret)
            #shear changes size, resize it back
            cropped_img = cropped_img.resize((int(box_w), int(box_h)), Image.BILINEAR)
            #paste back
            ret.paste(cropped_img, bbox)

        for y_info in self.along_y_info:
            M        = y_info["M"]
            new_h    = y_info["new_h"]
            color    = y_info["color"]
            anno_idx = y_info["anno_idx"]
            
            bbox = list(map(int, self.annotations[anno_idx]["bbox"]))
            box_w = bbox[2] - bbox[0]
            box_h = bbox[3] - bbox[1]
            
            #get box img to shear
            cropped_img = ret.crop(bbox)
            #do shear
            cropped_img = cropped_img.transform((int(box_w), int(new_h)), Image.AFFINE, M, resample=Image.BILINEAR, fillcolor=color)
            transformed_w, transformed_h = cropped_img.size
            if transformed_w<=0 or transformed_w<=0:
                return np.asarray(ret)
            #shear changes size, resize it back
            cropped_img = cropped_img.resize((int(box_w), int(box_h)), Image.BILINEAR)
            #paste back
            ret.paste(cropped_img, bbox)

        return np.asarray(ret)

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return NoOpTransform()


class BoxContrastTransform(WrapTransform):
    def __init__(self, invert_region):
        """Args
        invert_region(list): regions that computed to do invert
        """
        super().__init__()
        self._set_attributes(locals())
    
    def apply_image(self, img):
        ret = Image.fromarray(img)
        for region in self.invert_region:
            cropped_img = ret.crop(region)
            cropped_img = ImageOps.invert(cropped_img)
            ret.paste(cropped_img, region)

        return np.asarray(ret)
    
    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return NoOpTransform()


class NoiseTransform(WrapTransform):
    def __init__(self, rate):
        """Args
        rate(float): the rate pixles to be changed into noise
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        ret = Image.fromarray(img)
        img_w, img_h = ret.size
        noise_num  = int(self.rate * img_w * img_h)
        #add a random noise on a random position
        for _ in range(noise_num):
            x = random.randint(1, img_w-1)
            y = random.randint(1, img_h-1)
            noise = random.randint(0, 255)
            ret.putpixel((x,y), noise)
        
        return np.asarray(ret)
    
    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return NoOpTransform()


class BoxEraseTransform(WrapTransform):
    def __init__(self, fill_region):
        """Args:
        fill_region: regions to erase
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        ret = Image.fromarray(img)
        #fill the region
        for region in self.fill_region:
            ret = fill_region(ret, region)
        
        return np.asarray(ret)
    
    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def inverse(self):
        return NoOpTransform()


class MosaicTransform(WrapTransform):
    def __init__(self, mosaic_direction):
        """Args:
        mosaic_direction: directions to tile img, "right" or "bottom"
        """
        super().__init__()
        self._set_attributes(locals())
    
    def apply_image(self, img):
        has_mosaic_right  = False
        has_mosaic_bottom = False
        ret = Image.fromarray(img)

        for direction in self.mosaic_direction:
            mosaic_img = ret.copy()
            img_w, img_h = ret.size
            #right
            if direction=="right" and (not has_mosaic_right):
                ret = ImageOps.expand(ret, border=(0, 0, img_w, 0))
                ret.paste(mosaic_img, (img_w, 0))
                has_mosaic_right = True
            #bottom
            elif not has_mosaic_bottom:
                ret = ImageOps.expand(ret, border=(0, 0, 0, img_h))
                ret.paste(mosaic_img, (0, img_h))
                has_mosaic_bottom = True
        
        return np.asarray(ret)

    def apply_coords(self, coords):
        

        return coords

    def apply_annotations(self, annotations):
        category_ids = [anno["category_id"] for anno in annotations]

        has_mosaic_right  = False
        has_mosaic_bottom = False
        for direction in self.mosaic_direction:
            img_w, img_h = ret.size
            #right
            if direction=="right" and (not has_mosaic_right):
                #repeat coords
                coords = np.tile(coords, (2, 1))
                #compute new boxes for right side
                coords[coords.shape[0]//2:, 0] += img_w # x
                has_mosaic_right = True
            #bottom
            elif not has_mosaic_bottom:
                coords = np.tile(coords, (2, 1))
                coords[coords.shape[0]//2:, 1] += img_h # y
                has_mosaic_bottom = True

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation)
        return segmentation

    def inverse(self):
        return NoOpTransform()


class BoxMoveTransform(WrapTransform):
    def __init__(self, move_info):
        """Args:
        move_info: how to move box (rescale and paste point)
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        ret = Image.fromarray(img)

        for info in self.move_info:
            anno_id = info["anno_id"]
            anno_move_info = info["anno_move_info"]

            for single_move_info in anno_move_info:
                dst_size = single_move_info["dst_size"]
                to_paste_point = single_move_info["to_paste_point"]
                raw_box_coor = single_move_info["raw_box_coor"]

                cropped_img = ret.crop(raw_box_coor)
                cropped_img = cropped_img.resize(dst_size, Image.BILINEAR)
                to_paste_point = list(map(int, to_paste_point))
                ret.paste(cropped_img, to_paste_point)

        return np.asarray(ret)
    
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        #TODO, not implement yet
        return segmentation

    def apply_annotations(self, annotations):
        for info in self.move_info:
            anno_id = info["anno_id"]
            anno_move_info = info["anno_move_info"]

            for single_move_info in anno_move_info:
                anno = {}

                dst_size = single_move_info["dst_size"]
                to_paste_point = single_move_info["to_paste_point"]

                new_box_left = to_paste_point[0]
                new_box_top = to_paste_point[1]
                new_box_right = to_paste_point[0] + dst_size[0]
                new_box_bottom = to_paste_point[1] + dst_size[1]

                anno['bbox'] = [float(new_box_left), float(new_box_top), float(new_box_right), float(new_box_bottom)]
                anno['bbox_mode'] = annotations[anno_id]["bbox_mode"]
                anno['category_id'] = int(annotations[anno_id]["category_id"])

                annotations.append(anno)
        
        return annotations

    def inverse(self):
        return NoOpTransform()



def HFlip_rotated_box(transform, rotated_boxes):
    """
    Apply the horizontal flip transform on rotated boxes.

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    # Transform x_center
    rotated_boxes[:, 0] = transform.width - rotated_boxes[:, 0]
    # Transform angle
    rotated_boxes[:, 4] = -rotated_boxes[:, 4]
    return rotated_boxes


def Resize_rotated_box(transform, rotated_boxes):
    """
    Apply the resizing transform on rotated boxes. For details of how these (approximation)
    formulas are derived, please refer to :meth:`RotatedBoxes.scale`.

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    scale_factor_x = transform.new_w * 1.0 / transform.w
    scale_factor_y = transform.new_h * 1.0 / transform.h
    rotated_boxes[:, 0] *= scale_factor_x
    rotated_boxes[:, 1] *= scale_factor_y
    theta = rotated_boxes[:, 4] * np.pi / 180.0
    c = np.cos(theta)
    s = np.sin(theta)
    rotated_boxes[:, 2] *= np.sqrt(np.square(scale_factor_x * c) + np.square(scale_factor_y * s))
    rotated_boxes[:, 3] *= np.sqrt(np.square(scale_factor_x * s) + np.square(scale_factor_y * c))
    rotated_boxes[:, 4] = np.arctan2(scale_factor_x * s, scale_factor_y * c) * 180 / np.pi

    return rotated_boxes


HFlipTransform.register_type("rotated_box", HFlip_rotated_box)
ResizeTransform.register_type("rotated_box", Resize_rotated_box)

# not necessary any more with latest fvcore
NoOpTransform.register_type("rotated_box", lambda t, x: x)
