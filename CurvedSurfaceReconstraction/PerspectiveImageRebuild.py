# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:06:33 2023

@author: Ahmed H. Hanfy
"""
import os
import cv2
import sys
import glob
import screeninfo
import numpy as np
import matplotlib.pyplot as plt
from linedrawingfunctions import (
                                 XCheck, 
                                 InclinedLine,
                                 IncParameters,
                                 IntersectionPoint
                                 )

px = 1/plt.rcParams['figure.dpi']

# %% OpenCV colors
class CVColor:
    """
    A class to represent common colors used in OpenCV.
    This class provides RGB tuples for a variety of commonly used colors.

    Supported colors:
       BLACK, WHITE, RED, GREEN, BLUE, GREENBLUE, YELLOW, CYAN, MAGENTA,
       FUCHSIPINK, GRAY, ORANGE

   """
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    GREENBLUE = (255, 128, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    FUCHSIPINK = (255, 128, 255)
    GRAY = (128, 128, 128)
    ORANGE = (0, 128, 255)

class BCOLOR:  # For coloring the text in terminal
    """
    A class to represent ANSI escape sequences for coloring terminal text.
    This class provides various ANSI escape codes to color and style text in
    terminal output.

    Supported formats:
        - BGOKBLUE: background blue
        - BGOKCYAN: background cyan
        - OKCYAN: cyan text
        - BGOKGREEN: background green
        - OKGREEN: green text
        - WARNING: yellow background (warning)
        - FAIL: yellow text (fail)
        - ITALIC: italic text
        - UNDERLINE: underlined text
        - ENDC: reset all attributes.

    """
    BGOKBLUE = '\033[44m'
    BGOKCYAN = '\033[46m'
    OKCYAN = '\033[36m'
    BGOKGREEN = '\033[42m'
    OKGREEN = '\033[32m'
    WARNING = '\033[43m'
    FAIL = '\033[33m'
    ENDC = '\033[0m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

class PerspectiveImageRebuild:
    def __init__(self) -> None:
        pass

    def extract_coordinates(self, event: int,  # call event
                            x: int, y: int, flags: int,  # mouse current status
                            parameters: tuple[str, tuple[int]]) -> None:
        """
        Record starting (x, y) coordinates on left mouse button click and draw
        a line that crosses all over the image, storing it in a global
        variable. In case of horizontal or vertical lines, it takes the average
        between points.

        Drawing steps:
            1. Push the left mouse on the first point.
            2. Pull the mouse cursor to the second point.
            3. The software will draw a thick red line (indicating the mouse
               locations) and a green line indicating the Final line result.
            4. To confirm, press the left click anywhere on the image, or
               to delete the line, press the right click anywhere on the image.
            5. Press any key to proceed.

        Parameters:
            - event (int): The type of event (e.g., cv2.EVENT_LBUTTONDOWN).
            - x (int): The x-coordinate of the mouse cursor.
            - y (int): The y-coordinate of the mouse cursor.
            - flags (int): Flags associated with the mouse event.
            - parameters (tuple): A tuple containing:
                - Name of the window to display the image.
                - Image shape (tuple of y-length and x-length).

        Returns:
            None

        Example:
            >>> instance = SOA()
            >>> cv2.setMouseCallback(window_name, instance.extract_coordinates, parameters)

        .. note::
            - If 'Inc' is provided as the line type, it uses the 'InclinedLine' method
              to calculate the inclined line and display it on the image.

        """
        if   parameters[0] == "Z Projection line": color = CVColor.BLUE  # Blue
        elif parameters[0] == "Y Projection line": color = CVColor.GREEN # Green
        elif parameters[0] == "X Projection line": color = CVColor.RED   # Red
        else: color = CVColor.YELLOW  # ................................ Yellow

        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ClickCount += 1
            if len(self.TempLine) == 2:
                self.line_coordinates = self.TempLine
            elif len(self.TempLine) == 0: self.TempLine = [(x,y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            if len(self.TempLine) < 2:
                self.TempLine.append((x,y))

                # Draw temprary line for confirmation
                cv2.line(self.Temp, self.TempLine[0], self.TempLine[1], 
                         CVColor.ORANGE, 2)
                P1, P2, m, a = InclinedLine(self.TempLine[0], self.TempLine[1],
                                            imgShape = parameters[1])
                cv2.line(self.Temp, P1, P2, color, 1)
                cv2.imshow(parameters[0], self.Temp)

            elif self.ClickCount == 2:
                   
                self.Temp = self.clone.copy()
                cv2.imshow(parameters[0], self.clone)
                # storing the vertical line
                P1, P2, m, a = InclinedLine(self.line_coordinates[0],
                                            self.line_coordinates[1],
                                            imgShape = parameters[1])
                cv2.line(self.Temp, P1, P2, color, 1)
                avg = [P1, P2, m, a]

                self.Reference.append(avg)
                self.clone = self.Temp.copy()
                cv2.imshow(parameters[0], self.clone)
                print('registered line: {}'.format(avg))

        # Delete draw line before storing    
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.TempLine = []
            if self.ClickCount>0: self.ClickCount -= 1
            self.Temp = self.clone.copy()
            cv2.imshow(parameters[0], self.Temp)


    def LineDraw(self, img: np.ndarray[int],             # BG image
                 LineNameInd: int,                       # Line info.
                 Intialize=False, **kwargs) -> list:     # Other parameters
        """
        Drive the extract_coordinates function to draw lines.

        Parameters:
            - **img (numpy.ndarray)**: A single OpenCV image.
            - **LineNameInd (int)**: Index of the window title from the list.
            - **Initialize (bool, optional)**: To reset the values of Reference and line_coordinates for a new line set. True or False (Default: False).

        Returns:
            list: Cropping limits or (line set).

        Example:
            >>> instance = SOA()
            >>> line_set = instance.LineDraw(image, 'V', 0, Initialize=True)
            >>> print(line_set)

        .. note::
            - The function uses the `extract_coordinates` method to interactively draw lines on the image.
            - It waits until the user presses a key to close the drawing window.

        .. note::
           ``LineNameInd`` is the index number refering to one of these values as window title:

            0. "First Reference Line (left)",
            1. "Second Reference Line (right)",
            2. "Horizontal Reference Line",
            3. "estimated shock location"

        """

        self.clone = img.copy()
        self.Temp = self.clone.copy()
        self.TempLine = []
        self.ClickCount = 0
        # Window titles
        WindowHeader = ["Z Projection line",
                        "Y Projection line",
                        "Chord line"]

        if Intialize:
            self.Reference = []
            self.line_coordinates = []
        shp = img.shape
        # win_x, win_y = self.screenMidLoc(shp)

        prams = [WindowHeader[LineNameInd], shp]
            
        cv2.imshow(WindowHeader[LineNameInd], self.clone)
        cv2.setMouseCallback(WindowHeader[LineNameInd], 
                             self.extract_coordinates,prams)
        # Wait until user press some key
        cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
        return self.Reference
        # return self.clone