import io
import zipfile
from math import ceil
from pathlib import Path
from typing import Tuple, List
from urllib.request import urlretrieve

import PIL
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import rescale
from skimage.util import img_as_float


def __download__() -> None:
    """
    Downloads the workshop data from the instructor's shared box folder.
    """
    url = "https://github.com/RaphaelaHeil/clustering-dhnb/releases/download/v1.0/digits.zip"
    urlretrieve(url, "digits.zip")


def __extract__() -> None:
    """
    Extract the workshop archive "digits.zip" in the current directory.
    """
    with zipfile.ZipFile("digits.zip", "r") as zip_ref:
        zip_ref.extractall(".")


def retrieve_dhnb_images() -> None:
    """
    Retrieves and extracts the images for the image analysis part of the DHNB 2022 workshop "Introduction to
    Text and Image Analysis in Python".
    """
    __download__()
    __extract__()
    if Path("dida_single_digit_10k").exists():
        print("Successfully donwloaded dataset!")
    else:
        print("Something has gone wrong, please check in with a helper")


def read_all_images(base_path: Path = Path("dida_single_digit_10k"), images_per_digit: int = 10) -> Tuple[
    List[np.ndarray], List[int]]:
    """
    Loads the specified number of samples per digit from the workshop's image dataset.

    Parameters
    ----------
    base_path : Path
        path from which to load the images, default: "dida_single_digit_10k"
    images_per_digit : int
        number of samples to load per digit, default: 10

    Returns
    -------
    Tuple[List[np.ndarray], List[int]]
        List of images, as numpy arrays (intensity range [0,1]), and a list of corresponding digit labels

    """
    images = []
    labels = []
    image_names = list(base_path.glob("*.jpg"))
    image_names.sort(key=lambda x: int(x.stem.split("_")[0])*10 + int(x.stem.split("_")[1]) )
    max_images = images_per_digit * 10
    for filename in image_names[:max_images]:
        images.append(img_as_float(imread(filename)))
        labels.append(int(filename.stem.split("_")[1]))

    print("Loaded", len(images), "images from:", base_path)
    return images, labels


def scale_and_pad(image: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Scales the image to a given height, while maintaining the aspect ratio, and pads it to the specified width. Images
    that, after scaling, are wider than desired will be cut to obtain the target shape.

    Parameters
    ----------
    image : np.ndarray
        image to be processed
    shape : Tuple[int, int]
        desired output shape

    Returns
    -------
    np.ndarray
        rescaled and padded image

    """
    targetHeight, targetWidth = shape
    scale = targetHeight / image.shape[0]

    if image.ndim == 2:
        rescaled = rescale(image, scale)
        newHeight, newWidth = rescaled.shape
        if newWidth > targetWidth:
            newWidth = targetWidth
        paddedImage = np.ones(shape)
        paddedImage[:newHeight, :newWidth] = rescaled[:newHeight, :newWidth]
        return paddedImage
    elif image.ndim == 3:
        rescaled = rescale(image, (scale, scale, 1))
        newHeight, newWidth, _ = rescaled.shape
        if newWidth > targetWidth:
            newWidth = targetWidth
        paddedImage = np.ones(shape + (3,))
        paddedImage[:newHeight, :newWidth, :] = rescaled[:newHeight, :newWidth, :]
        return paddedImage
    else:
        raise ValueError(
                "Expected input to be two- or three-dimensional but received {} dimensions.".format(image.ndim))


def digit_grid(images: List[np.ndarray], ncols: int = 10, spacer: Tuple[int, int] = (5, 5),
               spacer_value: float = 1.0, scale_pad_shape:Tuple[int, int]=(64,100), figsize:Tuple[int,int]=(10,10)) -> np.ndarray:
    """
    Arranges the provided images in a grid of `ncols` columns. Pads the last row with empty spaces if the number of
    provided images is not sufficient to fill the entire row.
    If the images are of different sizes, scales the images to the height and pads to the width specified by `scale_pad_shape`.

    Parameters
    ----------
    images : List[np.ndarray]
        List of images to be formatted in a grid. **All images are expected to be of the same shape!**
    ncols : int
        Desired number of columns in the grid. Default: 10
    spacer : Tuple[int, int]
        Shape of the spacer to use between rows and columns. Default: (5,5)
    spacer_value : float
        Intensity value for the spacer. Default: 1.0
    scale_pad_shape : Tuple[int, int]
        Shape for scaling and padding, if `images` have varying shapes
    

    Returns
    -------
    np.ndarray
        Images arranged in a grid
    """
    
    if len(set([img.shape for img in images])) > 1:
        images = [scale_and_pad(img, scale_pad_shape) for img in images]
    
    
    col_spacer_height = images[0].shape[0]
    row_spacer_width = images[0].shape[1] * ncols + spacer[0] * (ncols - 1)

    nrows = ceil(len(images) / ncols)

    if images[0].ndim == 3:
        col_spacer = np.ones((col_spacer_height, spacer[0], 3)) * spacer_value
        row_spacer = np.ones((spacer[1], row_spacer_width, 3)) * spacer_value
        dummyImage = np.ones((col_spacer_height, images[0].shape[1], 3)) * spacer_value
    else:
        col_spacer = np.ones((col_spacer_height, spacer[0])) * spacer_value
        row_spacer = np.ones((spacer[1], row_spacer_width)) * spacer_value
        dummyImage = np.ones((col_spacer_height, images[0].shape[1])) * spacer_value

    image_index = 0
    accumulator = np.ndarray((0, 0))

    for row in range(nrows):
        rowAcc = np.ndarray((0, 0))
        for col in range(ncols):
            if image_index == len(images):
                # not enough images to fill the row, add dummyImage with spacer colour to get rectangular array
                rowAcc = np.hstack([rowAcc, col_spacer, dummyImage])
                continue  # keep on running until all columns are filled in
            if col == 0:
                rowAcc = images[image_index]
            else:
                rowAcc = np.hstack([rowAcc, col_spacer, images[image_index]])
            image_index += 1
        if row == 0:
            accumulator = rowAcc
        else:
            accumulator = np.vstack([accumulator, row_spacer, rowAcc])
            
    fig = plt.figure(figsize=figsize)
    if images[0].ndim == 2:
        plt.imshow(accumulator, cmap="gray")
    else:
        plt.imshow(accumulator)

def __to_display__(image: np.ndarray) -> bytes:
    """
    Converts the image into a byte format displayable by the `ipywidgets.Image` widget.

    Parameters
    ----------
    image : np.ndarray
        image to be converted for display

    Returns
    -------
    bytes
        the image in a displayable byte-format
    """
    img_byte_arr = io.BytesIO()
    PIL.Image.fromarray(image * 255).convert("L").save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()



class AnnotationView():
    
    def __init__(self, black_and_white_images, cluster_labels):
        """
        Creates a basic annotation view, consisting of one image and accompanying text input per row.

        Parameters
        ----------
        heatmaps : List[np.ndarray]
            List of images to be displayed alongside a text input.
        """
        
        self.cluster_labels = cluster_labels
        self.heatmaps=[]
        number_of_clusters = len(np.unique(cluster_labels))

        for cluster_id in range(number_of_clusters):
            selected = np.array(black_and_white_images)[cluster_labels==cluster_id]
            meanImage = np.mean(selected, axis=0)
            self.heatmaps.append(meanImage)
        
        self.text_widgets = []
        wid = []

        for h in self.heatmaps:
            a = widgets.Text(value="0", placeholder="Insert digit")
            self.text_widgets.append(a)
            box = widgets.HBox([widgets.Image(value=__to_display__(h)), a])
            wid.append(box)
        button = widgets.Button(description="Submit") 
        # XXX: a dummy button to give the visual clue that the annotation cell does not have to be re-run 
        # to "submit" the values. User-provided values should be retrieved using 'get_annotations'
        wid.append(button)
        self.vbox = widgets.VBox(wid)
        
    def get_annotations(self):
        annotation = []
        for i in self.text_widgets:
            annotation.append(int(i.value))
        return annotation
    
    def get_manual_labels(self):
        manual_labels = self.get_annotations()
        manual_classification = np.zeros_like(self.cluster_labels)
        for cluster_index in range(len(np.unique(self.cluster_labels))):
            digit_label = manual_labels[cluster_index]
            manual_classification[self.cluster_labels == cluster_index] = digit_label
        return manual_classification
    
    def show(self):
        display(self.vbox)

        
