# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:00:37 2024

@author: Ahmed H. Hanfy
"""
import sys
import numpy as np
from .PerspectiveImageRebuild import BCOLOR

def XCheck(x: float, Shp: tuple[int], slope: float, a: float) -> tuple[float]:
    """
    Check and calculate the image boundary y-coordinate based on the given x-coordinate and slope.

    This function takes an x-coordinate, image shape parameters (Shp), slope, and intercept (a) as inputs,
    and calculates the corresponding y-coordinate (p2) based on the specified conditions.

    Parameters:
        - **x (float)**: The x-coordinate to be checked.
        - **Shp (tuple)**: A tuple containing shape parameters (Shp[0] for y-axis limit, Shp[1] for x-axis limit).
        - **slope (float)**: The slope of the line.
        - **a (float)**: The y-intercept of the line.

    Returns:
        tuple: A tuple (p2) representing the calculated point (x, y).

    Example:
        >>> instance = PerspectiveImageRebuild()
        >>> result = instance.XCheck(2.5, (10, 5), 2, 3)
        >>> print(result)
        (2.5, 8)

    .. note::
        - If x is within the range [0, Shp[1]], the y-coordinate is calculated based on the line equation.
        - If x is greater than Shp[1], the y-coordinate is calculated at the point (Shp[1], y2), where y2 is determined by the line equation.
        - If x is less than 0, the y-coordinate is calculated at the point (0, y2), where y2 is determined by the line equation.

    """
    if   x >= 0 and x <= Shp[1]:                           p2 = (x, Shp[0])
    elif x >= 0 and x >  Shp[1]: y2 = int(Shp[1]*slope+a); p2 = (Shp[1],y2)
    elif x <  0 and x <= Shp[1]: y2 = int(a);              p2 = (0,y2)
    return p2

def InclinedLine(P1: tuple[int], P2: tuple[int] = (), 
                 slope: float = None, 
                 imgShape: tuple[int] = ()) -> tuple[tuple[int], 
                                                     tuple[int],float,float]:
    """
    Generates the inclined line equation from two points or one point and slope.

    The image boundary/shape should be given.

    Parameters:
        - **P1 (tuple)**: First point tuple (a1, b1).
        - **P2 (tuple, optional)**: Second point tuple (a2, b2). Defaults to ().
        - **slope (float, optional)**: Slope of the line. Defaults to None.
        - **imgShape (tuple)**: Image size (y-length, x-length).

    Returns:
        tuple: A tuple containing:
            - first boundary point tuple.
            - second boundary point tuple.
            - line slope.
            - y-intercept.

    Example:
        >>> result = InclinedLine((0, 0), (2, 4), imgShape=(5, 5))
        >>> print(result)
        ((0, 0), (5, 5), 1.0, 0)

    .. note::
        - If `imgShape` is not provided, the function prints an error message and aborts the program.
        - If only one point (`P1`) and slope (`slope`) are provided, the function calculates the second point.
        - If the line is not vertical or horizontal, it calculates the boundary points based on the image shape.
        - If the line is vertical, the slope is `np.inf`, and the function returns vertical boundary points.
        - If the line is horizontal, the slope is 0, and the function returns horizontal boundary points.

    """
    if len(imgShape) < 1: 
        print(f'{BCOLOR.FAIL}Error: {BCOLOR.ENDC}{BCOLOR.ITALIC}Image shape is not provided, program aborting ...{BCOLOR.ENDC}')
        sys.exit()
        
    if len(P2) > 0 and slope is None:
        dx = P1[0]-P2[0];   dy = P1[1]-P2[1]
        if dx != 0: slope = dy/dx
    elif len(P2) == 0 and slope is np.inf: dx = 0;
    else: dx = -1 
         
    if slope != 0 and slope is not None and slope is not np.inf:
        a = P1[1] - slope*P1[0]
        Xmax = int((imgShape[0]-a)/slope)
        Xmin = int(-a/slope)
        if   Xmin >= 0 and Xmin <= imgShape[1]:
            p1 = (Xmin,0)
            p2 = XCheck(Xmax,imgShape,slope,a)
        elif Xmin >= 0 and Xmin >  imgShape[1]:
            y = int(imgShape[1]*slope+a)
            p1 = (imgShape[1],y)
            p2 = XCheck(Xmax,imgShape,slope,a)
        else:
            y1 = int(a);
            p1 = (0,y1)
            p2 = XCheck(Xmax,imgShape,slope,a)
        return p1, p2, slope, a
    elif dx == 0:
        return (P1[0],0), (P1[0],imgShape[0]), np.Inf, 0 
    else:
        return (0,P1[1]), (imgShape[1],P1[1]), 0, P1[1]  

def IncParameters(p1: tuple[int] , p2: tuple[int]) -> tuple[float]:
    """
    Calculate the slope and intercept of the line passing through two points.

    :param p1: A tuple representing the coordinates (x, y) of the first point.
    :param p2: A tuple representing the coordinates (x, y) of the second point.
    :return: A tuple containing the slope and y-intercept of the line. 
             If the line is vertical, the slope is returned as 0 and 
             the y-intercept as infinity.
    
    :Example:

    >>> inc_parameters((0, 0), (1, 1))
    (1.0, 0.0)
    >>> inc_parameters((0, 0), (0, 1))
    (0.0, inf)
    """
     
    dx = p1[0]-p2[0]
    dy = p1[1]-p2[1]
    if dy != 0 and dx != 0:
        slope = dy/dx
        a = p1[1] - slope * p1[0]
        return slope, a
    elif dx == 0:
        return 0, np.Inf
    else:
        return 0, 0


def IntersectionPoint(self, M: list[float], A: list[float],
                          Ref: list[tuple, tuple]) -> tuple[tuple[int],
                                                            list[float]]:
        """
        Calculate the intersection point between two lines.

        Parameters:
            - **M (list)**: List containing slopes of the two lines.
            - **A (list)**: List containing y-intercepts of the two lines.
            - **Ref (list)**: List containing reference points for each line.

        Returns:
            tuple:
                - A tuple containing: Pint (tuple): Intersection point coordinates (x, y).
                - A list of angles of the lines in degrees.

        Example:
            >>> slopes = [0.5, -2]
            >>> intercepts = [2, 5]
            >>> references = [(0, 2), (0, 5)]
            >>> intersection, angles = IntersectionPoint(slopes, intercepts, references)
            >>> print(intersection, angles)

        .. note ::
            - The function calculates the intersection point and angles between two lines specified by their slopes and y-intercepts.
            - Returns the intersection point coordinates and angles of the lines in degrees.
        """
        theta1 = np.rad2deg(np.arctan(M[0]))
        theta2 = np.rad2deg(np.arctan(M[1]))

        Xint, Yint = None, None

        if theta1 != 0 and theta2 != 0 and theta1 - theta2 != 0:
            Xint = (A[1] - A[0]) / (M[0] - M[1])
            Yint = M[0] * Xint + A[0]
        elif theta1 == 0 and theta2 != 0:
            Yint = Ref[0][1]             # Xint = Ref[0]
            Xint = (Yint - A[1]) / M[1]  # Yint = M[1]*Xint + A[1]
        elif theta2 == 0 and theta1 != 0:
            Xint = Ref[1][0]             # Xint = Ref[1]
            Yint = M[0] * Xint + A[0]    # Yint = M[0]*Xint + A[0]
        else:
            print(f'{BCOLOR.WARNING}Warning:{BCOLOR.ENDC}{BCOLOR.ITALIC}Lines are parallel{BCOLOR.ENDC}')

        Pint = (round(Xint), round(Yint))
        return Pint, [theta1, theta2]

def AngleFromSlope(slope: float) -> float:
    """
    Calculate the angle in degrees from the given slope.
    This function computes the angle in degrees corresponding to the provided slope value.

    Parameters:
        - **slope (float)**:   The slope of the line.

    Returns:
        float: The angle in degrees corresponding to the given slope.
        
    Example:
        >>> slope = 2
        >>> angle = AngleFromSlope(slope)
        >>> print(angle)
        26.56505117707799

    """
    if   slope > 0:ang_deg = 180 - np.rad2deg(np.arctan(slope))
    elif slope < 0:ang_deg = abs(np.rad2deg(np.arctan(slope)))
    else:  ang_deg = 90
    return ang_deg
