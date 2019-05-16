# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-

import cv2
import numpy

def get_face_data(image_file, flag):
    """
    Returns all facial landmarks in a given image.
    ______________________________________________
    Args:
        1. `image_file`:
            Either of three options:\n
                a. (int) Image data after being read with cv2.imread()\n
                b. File path of locally stored image file.\n
                c. Byte stream being received over multipart network request.\n\n
        2. `flag`:
            Used to denote the type of image_file parameter being passed.
            Possible values are IMG_DATA, FILE_READ, NETWORK_BYTE_STREAM respectively.
            By default its value is IMAGE_DATA, and assumes imread() image is passed.

    Returns:
        String with list of detected points of lips.

    Error:
        Returns `None` if face not found in image.

    """
    image = 0
    if flag == 'FILE_READ':
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif flag == 'NETWORK_BYTE_STREAM':
        image = cv2.imdecode(
            numpy.fromstring(image_file.read(), numpy.uint8), cv2.IMREAD_UNCHANGED
        )
    elif flag == 'IMAGE_DATA 'or flag is None:
        image = image_file
    landmarks = __get_landmarks(image)
    if landmarks[0] is None or landmarks[1] is None:
        return None
    return landmarks


def get_lips(image_file, flag=None):
    """
    Returns points for lips in given image.
    _______________________________________
    Args:
        1. `image_file`:
            Either of three options:\n
                a. (int) Image data after being read with cv2.imread()\n
                b. File path of locally stored image file.\n
                c. Byte stream being received over multipart network reqeust.\n\n
        2. `flag`:
            Used to denote the type of image_file parameter being passed.
            Possible values are IMG_DATA, FILE_READ, NETWORK_BYTE_STREAM respectively.
            By default its value is IMAGE_DATA, and assumes imread() image is passed.

    Returns:
        String with list of detected points of lips.

    Error:
        Returns `None` if face not found in image.

    """
    landmarks = get_face_data(image_file, flag)
    if landmarks is None:
        return None
    lips = ""
    for point in landmarks[48:]:
        lips += str(point).replace('[', '').replace(']', '') + '\n'
    return lips

if __name__ == '__main__':
    im = cv2.imread('test.png')
    lips = get_lips(im)