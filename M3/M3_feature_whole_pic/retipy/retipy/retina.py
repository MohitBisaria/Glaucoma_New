# Retipy - Retinal Image Processing on Python
# Copyright (C) 2017-2018  Alejandro Valdes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""retina module to handle basic image processing on retinal images"""

import base64
import math
import numpy as np
import warnings
from copy import copy
from io import BytesIO
from ..function_ import thinning
from matplotlib import pyplot as plt
from os import path
from PIL import Image
from scipy import ndimage
from skimage import color, feature, filters, io
from skimage.morphology import skeletonize
import cv2
import pandas as pd

import torch
import os
import sys
import torch

import sys
import os
import torch

# Correct Python path to M2_Artery_vein folder
#M2_PATH = "/home/mu/Desktop/Project_Code/M2_Artery_vein"
#if M2_PATH not in sys.path:
#    sys.path.append(M2_PATH)

#from .import model
#from .import utils

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
#artery_model = model.UNet(n_channels=3, n_classes=1).to(device)
#vein_model = model.UNet(n_channels=3, n_classes=1).to(device)

# Load weights using correct path
#artery_model.load_state_dict(torch.load(os.path.join(M2_PATH, "CP_best_F1_A.pth"), map_location=device))
#vein_model.load_state_dict(torch.load(os.path.join(M2_PATH, "CP_best_F1_V.pth"), map_location=device))

#artery_model.eval()
#vein_model.eval()



#import os
#print("\nRETINA FILE LOCATION:")
#print(os.path.abspath(__file__))

#print("\nPARENT FOLDER:")
#print(os.path.abspath(os.path.dirname(__file__)))

#print("\nALL PARENTS:")
#print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
#print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
#print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

class Retina(object):
    """
    Retina class that internally contains a matrix with the image data for a retinal image

    :param image: a numpy array with the image data
    :param image_path: path to an image to be open
    """

    @staticmethod
    def _open_image(img_path):
        return cv2.resize(io.imread(img_path), dsize=(912, 912), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def get_base64_image(image: np.ndarray):
        """Converts numpy image (RGB or grayscale) to base64 string."""
        if len(image.shape) == 2:  # Grayscale
            temp_image = Image.fromarray(image.astype('uint8'), 'L')
        elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB/BGR color
            temp_image = Image.fromarray(image.astype('uint8'), 'RGB')
        else:
            raise ValueError(f"Unsupported image shape for base64 conversion: {image.shape}")

        buffer = BytesIO()
        temp_image.save(buffer, format="png")
        return str(base64.b64encode(buffer.getvalue()).decode('utf-8'))

    def __init__(self, image: np.ndarray, image_path: str, store_path: str):
        if image is None:
            self.np_image = self._open_image(image_path)
            self.segmentation_path = store_path + image_path.split('_skeleton')[1]
            self.vessel_image = self._open_image(self.segmentation_path)
            _, file = path.split(image_path)
            self._file_name = file
        else:
            self.np_image = image
            self._file_name = image_path
            self.segmentation_path = store_path

            # If image is already passed as numpy (from Streamlit), use it directly
            if isinstance(image, np.ndarray):
                self.vessel_image = image
            else:
                self.vessel_image = self._open_image(self.segmentation_path)

        # Resolution: try to read from crop_info.csv, otherwise default to 1.0
        try:
            if store_path and path.exists(store_path):
                csv_path = path.join(store_path.split('M2')[0], 'M0', 'crop_info.csv')
                if path.exists(csv_path):
                    resolution_list = pd.read_csv(csv_path)
                    self.resolution = resolution_list['Scale_resolution'][
                        resolution_list['Name'] == self._file_name
                    ].values[0]
                else:
                    self.resolution = 1.0
            else:
                self.resolution = 1.0
        except Exception:
            self.resolution = 1.0

        self.segmented = False
        self.old_image = None
        self.original_base64 = self.get_base64_image(self.np_image)
        self.depth = 1
        self.shape = self.np_image.shape
        self._disc_cache = None  # cache disc/cup estimation

    ##################################################################################################
    # Vessel segmentation & basic derived metrics
    ##################################################################################################

    def extract_vessels(self):
        """
        Basic vessel segmentation using grayscale conversion, CLAHE, Otsu thresholding,
        and morphological cleanup. Compatible with numpy images.
        """
        img = self.np_image.copy()

        # Convert to grayscale if RGB/BGR
        if len(img.shape) == 3:
            # handle both RGB and BGR reasonably
            try:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img.astype(np.uint8)

        # CLAHE to enhance vessel contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)

        # Otsu threshold
        _, vessel_mask = cv2.threshold(
            img_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Morphological cleanup (remove noise & small objects)
        vessel_mask = cv2.medianBlur(vessel_mask, 5)
        kernel = np.ones((3, 3), np.uint8)
        vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Convert to binary (0 and 1)
        return (vessel_mask > 0).astype(np.uint8)

    def skeletonize_mask(self, mask):
        """
        Applies skeletonization to a binary vessel mask and returns the thinned skeleton.
        """
        if mask is None:
            return None

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        binary_mask = mask > 0
        skel = skeletonize(binary_mask)
        return skel.astype(np.uint8)

    def extract_artery_mask(self):
        """
        Returns a binary mask of arteries using the classification() function if available.
        If classification fails, returns None.
        """
        vessel_mask = self.extract_vessels()
        try:
            from . import vessel_classification
            av_mask = vessel_classification.classification(self.np_image, vessel_mask)
            artery_mask = (av_mask == 1).astype(np.uint8)  # assuming 1 = artery
            if artery_mask.sum() == 0:
                return None
            return artery_mask
        except Exception:
            return None

    def extract_vein_mask(self):
        """
        Returns a binary mask of veins using the classification() function if available.
        If classification fails, returns None.
        """
        vessel_mask = self.extract_vessels()
        try:
            from . import vessel_classification
            av_mask = vessel_classification.classification(self.np_image, vessel_mask)
            vein_mask = (av_mask == 2).astype(np.uint8)  # assuming 2 = vein
            if vein_mask.sum() == 0:
                return None
            return vein_mask
        except Exception:
            return None

    def vessel_density(self, vessel_type="all"):
        """
        Computes vessel density for arteries, veins, or all vessels.
        vessel_type: "artery", "vein", or "all"
        """
        try:
            vessel_mask = self.extract_vessels()  # full vessel mask (binary)

            # If artery/vein segmentation is available
            if vessel_type.lower() == "artery":
                vessel_mask = self.extract_artery_mask()
            elif vessel_type.lower() == "vein":
                vessel_mask = self.extract_vein_mask()

            if vessel_mask is None:
                return np.nan

            total_pixels = vessel_mask.size
            vessel_pixels = np.sum(vessel_mask > 0)

            if total_pixels == 0:
                return np.nan

            return vessel_pixels / total_pixels  # ratio 0–1

        except Exception:
            return np.nan

    def average_width(self):
        """
        Estimates average vessel width using:
        width ≈ vessel_mask_area / skeleton_length
        """
        try:
            vessel_mask = self.extract_vessels()                  # Full vessel mask
            skeleton = self.skeletonize_mask(vessel_mask)         # Thin skeleton

            if skeleton is None:
                return np.nan

            vessel_area = np.sum(vessel_mask > 0)                 # Total vessel pixels
            skeleton_length = np.sum(skeleton > 0)                # 1-pixel-wide vessels

            if skeleton_length == 0:
                return np.nan

            avg_width = vessel_area / skeleton_length
            return float(round(avg_width, 3))

        except Exception:
            return np.nan
        

    def extract_artery_mask(self):
        """Returns binary artery segmentation mask"""
        input_tensor = preprocess_image(self.np_image).to(device)
        with torch.no_grad():
           pred = artery_model(input_tensor)
        mask = pred.squeeze().cpu().numpy() > 0.5
        return mask.astype(np.uint8)

    def extract_vein_mask(self):
        """Returns binary vein segmentation mask"""
        input_tensor = preprocess_image(self.np_image).to(device)
        with torch.no_grad():
           pred = vein_model(input_tensor)
        mask = pred.squeeze().cpu().numpy() > 0.5
        return mask.astype(np.uint8)


    ##################################################################################################
    # Optic disc / cup & CDR (approximate, heuristic)
    ##################################################################################################

    def _estimate_disc_region(self):
        """
        Heuristic estimation of optic disc (bright circular region) and an inner cup.

        Returns (cx, cy, disc_radius, cup_radius) in pixels.
        """
        if self._disc_cache is not None:
            return self._disc_cache

        img = self.np_image
        if img.ndim == 3:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.astype(np.uint8)

        h, w = gray.shape[:2]

        # Smooth and enhance bright region
        gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
        gray_norm = cv2.normalize(gray_blur, None, 0, 255, cv2.NORM_MINMAX)

        # Threshold high intensities (disc is brightest area)
        _, th = cv2.threshold(
            gray_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Get largest bright blob
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            (x, y), disc_r = cv2.minEnclosingCircle(c)
            cx, cy = int(x), int(y)
        else:
            # fallback to center of image with heuristic radius
            cx, cy = w // 2, h // 2
            disc_r = 0.12 * min(h, w)

        # approximate cup radius as a fraction of disc radius
        cup_r = 0.6 * disc_r

        self._disc_cache = (cx, cy, float(disc_r), float(cup_r))
        return self._disc_cache

    def disc_height(self):
        """Approximate optic disc height in pixels."""
        _, _, disc_r, _ = self._estimate_disc_region()
        return 2.0 * disc_r

    def disc_width(self):
        """Approximate optic disc width in pixels."""
        _, _, disc_r, _ = self._estimate_disc_region()
        return 2.0 * disc_r

    def cup_height(self):
        """Approximate optic cup height in pixels."""
        _, _, _, cup_r = self._estimate_disc_region()
        return 2.0 * cup_r

    def cup_width(self):
        """Approximate optic cup width in pixels."""
        _, _, _, cup_r = self._estimate_disc_region()
        return 2.0 * cup_r

    def cdr_vertical(self):
        """Vertical Cup-to-Disc Ratio (approximate)."""
        _, _, disc_r, cup_r = self._estimate_disc_region()
        if disc_r <= 0:
            return np.nan
        return float(cup_r / disc_r)

    def cdr_horizontal(self):
        """Horizontal Cup-to-Disc Ratio (approximate)."""
        # same as vertical under circular approximation
        return self.cdr_vertical()

    # aliases used elsewhere
    def vcdr(self):
        return self.cdr_vertical()

    def hcdr(self):
        return self.cdr_horizontal()

    ##################################################################################################
    # Image Processing functions (original code)
    ##################################################################################################

    def threshold_image(self):
        """Applies a thresholding algorithm to the contained image."""
        threshold = filters.threshold_mean(self.np_image)
        self.np_image = self.np_image > threshold
        self.depth = 1

    def detect_edges_canny(self, min_val=0, max_val=1):
        """
        Applies canny edge detection to the contained image. Fine tuning of the algorithm can be
        done using min_val and max_val
        """
        self._copy()
        self.np_image = feature.canny(self.np_image, low_threshold=min_val, high_threshold=max_val)

    def apply_thinning(self):
        """Applies a thinning algorithm on the stored image"""
        self._copy()
        self.np_image = thinning.thinning_zhang_suen(self.np_image)

    def erode(self, times):
        """
        Erodes the stored image
        :param times: number of times that the image will be eroded
        """
        self._copy()
        self.np_image = ndimage.binary_erosion(self.np_image, iterations=times)

    def dilate(self, times):
        """
        dilates the stored image
        :param times: number of times that the image will be dilated
        """
        self._copy()
        self.np_image = ndimage.binary_dilation(self.np_image, iterations=times)

    def reshape_square(self):
        """
        This function will normalise the image size, making a square with it and rounding the pixels:
        If the given image is 571 560, the new size will be 572 572, with zeroes in all new pixels.
        """
        max_value = self.shape[0] if self.shape[0] > self.shape[1] else self.shape[1]
        max_value = max_value + (max_value % 2)
        self.np_image = np.pad(
            self.np_image,
            ((0, max_value - self.shape[0]), (0, max_value - self.shape[1])),
            'constant',
            constant_values=(0, 0))
        self.shape = self.np_image.shape

    def reshape_by_window(self, window: int, is_percentage: bool = False) -> int:
        """
        Reshapes the internal image to be able to be divided by the given window size
        :param window: an integer with the window size. Considered as a square
        :param is_percentage: sets if the given window is a percentage of the image or a pixel value

        :return: the dimension of the image on which was resized.
        """
        dimension = window
        if is_percentage:
            # get the smallest dimension
            selected_dimension = self.shape[0] if self.shape[0] < self.shape[1] else self.shape[1]
            dimension = int(math.floor(selected_dimension / window))
            # make it even
            dimension += dimension % 2
            x_pixels = (math.ceil(self.shape[0] / dimension) * dimension) - self.shape[0]
            y_pixels = (math.ceil(self.shape[1] / dimension) * dimension) - self.shape[1]
        else:
            x_pixels = (math.ceil(self.shape[0] / window) * window) - self.shape[0]
            y_pixels = (math.ceil(self.shape[1] / window) * window) - self.shape[1]
        self.np_image = np.pad(
            self.np_image, ((0, x_pixels), (0, y_pixels)), 'constant', constant_values=(0, 0))
        self.shape = self.np_image.shape

        return dimension

    def get_window_sizes(self):
        """
        Returns an array with the possible window size that this image can be divided by without leaving empty space.
        584x584 would return [292,146,73]
        This is only available for square images (you can use reshape_square() before calling this method)
        :return: a list of possible window sizes.
        """
        sizes = []
        if self.shape[0] == self.shape[1]:
            current_value = self.shape[0]
            while current_value % 2 == 0:
                current_value = current_value // 2
                sizes.append(current_value)
        return sizes

    def skeletonization(self):
        """Applies a skeletonization algorithm to the contained image."""
        self.np_image = skeletonize(self.np_image)

    def bin_to_bgr(self):
        """Transform the image to a ndarray with depth:3"""
        h, w = self.np_image.shape
        image_bgr = np.zeros((h, w, 3))
        image_bgr[:, :, 0] = self.np_image
        image_bgr[:, :, 1] = self.np_image
        image_bgr[:, :, 2] = self.np_image
        self.np_image = image_bgr

    def get_uint_image(self):
        """
        Returns the np_image converted to uint8 and multiplied by 255 to simulate grayscale
        :return: a ndarray image
        """
        image = self.np_image.astype(np.uint8) * 255
        return image

    def reshape_for_landmarks(self, size: int):
        """
        Reshapes the internal image to fix errors with retinal images without borders
        :param size: an integer with the border size.
        """
        self.np_image = np.pad(self.np_image, pad_width=size, mode='constant', constant_values=0)
        self.shape = self.np_image.shape

    ##################################################################################################
    # I/O functions
    ##################################################################################################

    def _copy(self):
        self.old_image = copy(self.np_image)

    def undo(self):
        """Reverts the latest modification to the internal image."""
        self.np_image = self.old_image

    @property
    def filename(self):
        """Returns the filename of the retina image."""
        return self._file_name

    def _output_filename(self):
        return "/out_" + self.filename

    def save_image(self, output_folder):
        """Saves the image in the given output folder, the name will be out_<original_image_name>"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(output_folder + self._output_filename(), self.np_image)

    def view(self):  # pragma: no cover
        """show a window with the internal image"""
        io.imshow(self.np_image)
        plt.show()

    def compare_with(self, retinal_image):
        """
        Returns the difference between the given image and the stored one.

        :param retinal_image: the image to compare with
        :return:  a new Retina object with the differences
        """
        return Retina(self.np_image - retinal_image.np_image, "diff" + self.filename)




class Window(Retina):
    """
    a ROI (Region of Interest) that extends the Retina class
    TODO: Add support for more than depth=1 images (only if needed)
    """
    def __init__(self, image: Retina, dimension, method="separated", min_pixels=10):
        super(Window, self).__init__(
            image.np_image,
            image.filename,
            image.segmentation_path)
        self.windows, self.w_pos = Window.create_windows(image, dimension, method, min_pixels)
        if len(self.windows) == 0:
            raise ValueError("No windows were created for the given retinal image")
        self.shape = self.windows.shape
        self._mode = self.mode_pytorch
        self._tags = None

    @property
    def mode_pytorch(self):
        return "PYT"

    @property
    def mode_tensorflow(self):
        return "TF"

    @property
    def tags(self) -> np.ndarray:
        return self._tags

    @tags.setter
    def tags(self, value: np.ndarray):
        self._tags = value
        if value.shape[0] != self.shape[0]:
            raise ValueError(
                "Wrong set of tags, expected {} got {}".format(self.shape[0], value.shape[0]))

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode):
        """
        Changes the internal window ordering depending on the given mode.
        Tensorflow style is [batch, width, height, depth]
        Pytorch style is [batch, depth, width, height]
        :param mode: new mode to change, can be self.tensorflow or self.pytorch
        """
        if mode == self.mode_pytorch and self._mode == self.mode_tensorflow:
            twin = np.swapaxes(self.windows, 2, 3)
            self.windows = np.swapaxes(twin, 1, 2)
            self.shape = self.windows.shape
            self._mode = self.mode_pytorch
        elif mode == self.mode_tensorflow and self._mode == self.mode_pytorch:
            twin = np.swapaxes(self.windows, 1, 2)
            self.windows = np.swapaxes(twin, 2, 3)
            self.shape = self.windows.shape
            self._mode = self.mode_tensorflow

    def _window_filename(self, window_id):
        return "out_w" + str(window_id) + "_" + self.filename

    def save_window(self, window_id, output_folder):
        """
        Saves the specified window in the given output folder, the name will be out_<original_image_name>
        :param window_id: the window_id to save
        :param output_folder: destination folder
        """
        if window_id >= self.windows.shape[0]:
            raise ValueError(
                "Window value '{}' is more than allowed ({})".format(
                    window_id, self.windows.shape[0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(
                output_folder + self._window_filename(window_id), self.windows[window_id, 0])

    class _WindowIterator:
        def __init__(self, window: np.ndarray):
            self.__window = window
            self.__iterator = 0

        def __len__(self):
            return self.__window.shape[0]

        def __next__(self):
            self.__iterator += 1
            if self.__iterator <= self.__len__():
                return self.__window[self.__iterator - 1]
            else:
                raise StopIteration

    def __iter__(self):
        return self._WindowIterator(self.windows)

    @staticmethod
    def _create_tag_image(dx: int, dy: int, tags: list) -> np.ndarray:
        image = np.empty((dx, dy))
        boxes = len(tags) % 2 + len(tags)
        box_count = (boxes // 2)
        tag_x = dx // box_count
        tag_y = dy // 2
        for i in range(0, boxes):
            cx = (i % box_count) * tag_x
            cy = (i // box_count) * tag_y
            image[cx:cx+tag_x, cy:cy+tag_y] = tags[i]
        return image

    def view_window(self, w_id, layer):  # pragma: no cover
        io.imshow(self.windows[w_id, layer])
        plt.show()

    def set_tag_layer(self):
        """
        this method adds the tags values as a new depth window.
        The image will be splitted by the number of tags, resulting in a rectangle per
        tag which will contain the tag value repeated on it.
        """
        if self.tags is None:
            raise ValueError("tags is not set")
        if self.mode != self.mode_pytorch:
            self.mode = self.mode_pytorch

        for i in range(0, self.tags.shape[0]):
            self.windows[i, -1] = self._create_tag_image(
                self.windows.shape[2], self.windows.shape[3], self.tags[i])

    @staticmethod
    def create_windows(
            image: Retina, dimension, method="separated", min_pixels=10) -> tuple:
        """
        Creates multiple square windows of the given dimension for the current retinal image.
        Empty windows (i.e. only background) will be ignored

        Separated method will create windows of the given dimension size, that does not share any
        pixel, combined will make windows advancing half of the dimension, sharing some pixels
        between adjacent windows.
        :param image: an instance of Retina, to be divided in windows
        :param dimension:  window size (square of [dimension, dimension] size)
        :param method: method of separation (separated or combined)
        :param min_pixels: ignore windows with less than min_pixels with value.
                           Set to zero to add all windows
        :return: a tuple with its first element as a numpy array with the structure
                 [window, depth, height, width] and its second element as [window, 2, 2]
                 with the window position
        """

        if image.shape[0] % dimension != 0 or image.shape[1] % dimension != 0:
            raise ValueError(
                "image shape is not the same or the dimension value does not divide the image "
                "completely: sx:{} sy:{} dim:{}".format(image.shape[0], image.shape[1], dimension))

        #                      window_count
        windows = []
        windows_position = []
        window_id = 0
        img_dimension = image.shape[0] if image.shape[0] > image.shape[1] else image.shape[1]
        if method == "separated":
            windows = np.empty(
                [(img_dimension // dimension) ** 2, image.depth, dimension, dimension])
            windows_position = np.empty([(img_dimension // dimension) ** 2, 2, 2], dtype=np.int32)
            for x in range(0, image.shape[0], dimension):
                for y in range(0, image.shape[1], dimension):
                    cw = windows_position[window_id]
                    cw[0, 0] = x
                    cw[1, 0] = x + dimension
                    cw[0, 1] = y
                    cw[1, 1] = y + dimension
                    t_window = image.np_image[cw[0, 0]:cw[1, 0], cw[0, 1]:cw[1, 1]]
                    if t_window.sum() >= min_pixels:
                        windows[window_id, 0] = t_window
                        window_id += 1
        elif method == "combined":
            new_dimension = dimension // 2
            windows = np.empty(
                [(img_dimension // new_dimension) ** 2, image.depth, dimension, dimension])
            windows_position = np.empty([(img_dimension // new_dimension) ** 2, 2, 2], dtype=np.int32)
            if image.shape[0] % new_dimension != 0:
                raise ValueError(
                    "Dimension value '{}' is not valid, choose a value that its half value can split the image evenly"
                    .format(dimension))
            for x in range(0, image.shape[0] - new_dimension, new_dimension):
                for y in range(0, image.shape[1] - new_dimension, new_dimension):
                    cw = windows_position[window_id]
                    cw[0, 0] = x
                    cw[1, 0] = x + dimension
                    cw[0, 1] = y
                    cw[1, 1] = y + dimension
                    t_window = image.np_image[cw[0, 0]:cw[1, 0], cw[0, 1]:cw[1, 1]]
                    if t_window.sum() >= min_pixels:
                        windows[window_id, 0] = t_window
                        window_id += 1
        if window_id <= windows.shape[0]:
            if window_id == 0:
                windows = []
                windows_position = []
            else:
                windows = np.resize(
                    windows, [window_id, windows.shape[1], windows.shape[2], windows.shape[3]])
                windows_position = np.resize(windows_position, [window_id, 2, 2])

        #  print('created ' + str(window_id) + " windows")
        return windows, windows_position


def detect_vessel_border(image: Retina, ignored_pixels=1):
    """
    Extracts the vessel border of the given image, this method will try to extract all vessel
    borders that does not overlap.

    Returns a list of lists with the points of each vessel.

    :param image: the retinal image to extract its vessels
    :param ignored_pixels: how many pixels will be ignored from borders.
    """

    def neighbours(pixel, window):  # pragma: no cover
        """
        Creates a list of the neighbouring pixels for the given one. It will only
        add to the list if the pixel has value.

        :param pixel: the pixel position to extract its neighbours
        :param window:  the window with the pixels information
        :return: a list of pixels (list of tuples)
        """
        x_less = max(0, pixel[0] - 1)
        y_less = max(0, pixel[1] - 1)
        x_more = min(window.shape[0] - 1, pixel[0] + 1)
        y_more = min(window.shape[1] - 1, pixel[1] + 1)

        active_neighbours = []

        if window.np_image[x_less, y_less] > 0:
            active_neighbours.append([x_less, y_less])
        if window.np_image[x_less, pixel[1]] > 0:
            active_neighbours.append([x_less, pixel[1]])
        if window.np_image[x_less, y_more] > 0:
            active_neighbours.append([x_less, y_more])
        if window.np_image[pixel[0], y_less] > 0:
            active_neighbours.append([pixel[0], y_less])
        if window.np_image[pixel[0], y_more] > 0:
            active_neighbours.append([pixel[0], y_more])
        if window.np_image[x_more, y_less] > 0:
            active_neighbours.append([x_more, y_less])
        if window.np_image[x_more, pixel[1]] > 0:
            active_neighbours.append([x_more, pixel[1]])
        if window.np_image[x_more, y_more] > 0:
            active_neighbours.append([x_more, y_more])

        return active_neighbours
    
    
    def intersection(mask,image, it_x, it_y):
        """
        Remove the intersection in case the whole vessel is too long
        """
        vessel_ = image.np_image
        x_less = max(0, it_x - 1)
        y_less = max(0, it_y - 1)
        x_more = min(vessel_.shape[0] - 1, it_x + 1)
        y_more = min(vessel_.shape[1] - 1, it_y + 1)

        active_neighbours = (vessel_[x_less, y_less]>0).astype('float')+ \
                            (vessel_[x_less, it_y]>0).astype('float')+ \
                            (vessel_[x_less, y_more]>0).astype('float')+ \
                            (vessel_[it_x, y_less]>0).astype('float')+ \
                            (vessel_[it_x, y_more]>0).astype('float')+ \
                            (vessel_[x_more, y_less]>0).astype('float')+ \
                            (vessel_[x_more, it_y]>0).astype('float')+ \
                            (vessel_[x_more, y_more]>0).astype('float')

        if active_neighbours > 2:
            cv2.circle(mask,(it_y,it_x),radius=1,color=(0,0,0),thickness=-1)
        

        return mask,active_neighbours        
        
    '''
    # original remove x duplicate
    def vessel_extractor(window, start_x, start_y):
        """
        Extracts a vessel using adjacent points, when each point is extracted is deleted from the
        original image
        & Measure width
        """
        vessel = []
        width_list = []
        width_mask = np.zeros((window.np_image.shape))
        pending_pixels = [[start_x, start_y]]
        while pending_pixels:
            pixel = pending_pixels.pop(0)
            if window.np_image[pixel[0], pixel[1]] > 0:
                vessel.append(pixel)
                window.np_image[pixel[0], pixel[1]] = 0

                # add the neighbours with value to pending list:
                pending_pixels.extend(neighbours(pixel, window))

        # sort by x position
        vessel.sort(key=lambda item: item[0])

        # remove all repeating x values???????????
        current_x = -1
        filtered_vessel = []
        for pixel in vessel:
            if pixel[0] == current_x:
                pass
            else:
                filtered_vessel.append(pixel)
                current_x = pixel[0]

        vessel_x = []
        vessel_y = []
        for pixel in filtered_vessel:
            vessel_x.append(pixel[0])
            vessel_y.append(pixel[1])
            

        return [vessel_x, vessel_y]
    '''
    
    # 2021/10/31 remove setting of the sort & x duplication
    def vessel_extractor(window, start_x, start_y):
        """
        Extracts a vessel using adjacent points, when each point is extracted is deleted from the
        original image
        & Measure width
        """
        vessel = []
        width_list = []
        width_mask = np.zeros((window.np_image.shape))
        pending_pixels = [[start_x, start_y]]
        while pending_pixels:
            pixel = pending_pixels.pop(0)
            if window.np_image[pixel[0], pixel[1]] > 0:
                vessel.append(pixel)
                window.np_image[pixel[0], pixel[1]] = 0

                # add the neighbours with value to pending list:
                pending_pixels.extend(neighbours(pixel, window))

        # sort by x position
        '''
        vessel.sort(key=lambda item: item[0])

        # remove all repeating x values???????????
        current_x = -1
        filtered_vessel = []
        for pixel in vessel:
            if pixel[0] == current_x:
                pass
            else:
                filtered_vessel.append(pixel)
                current_x = pixel[0]
        '''
        filtered_vessel = vessel
        vessel_x = []
        vessel_y = []
        for pixel in filtered_vessel:
            vessel_x.append(pixel[0])
            vessel_y.append(pixel[1])
            

        return [vessel_x, vessel_y]
    
    
    vessels = []
    active_neighbours_list = []
    width_list_all = []
    mask_ = np.ones((image.np_image.shape))
    
    for it_x in range(ignored_pixels, image.shape[0] - ignored_pixels):
        for it_y in range(ignored_pixels, image.shape[1] - ignored_pixels):
            if image.np_image[it_x, it_y] > 0:
                mask,active_neighbours = intersection(mask_,image, it_x, it_y)
                active_neighbours_list.append(active_neighbours)
    
    image.np_image = image.np_image * mask
    
    #cv2.imwrite('./intersection_test/{}.png'.format(image._file_name),image.np_image)
    
    for it_x in range(ignored_pixels, image.shape[0] - ignored_pixels):
        for it_y in range(ignored_pixels, image.shape[1] - ignored_pixels):
            if image.np_image[it_x, it_y] > 0:
                vessel = vessel_extractor(image, it_x, it_y)
                vessels.append(vessel)
    
                
    return vessels
