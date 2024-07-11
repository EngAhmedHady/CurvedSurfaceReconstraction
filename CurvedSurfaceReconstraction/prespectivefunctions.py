# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:18:37 2024

@author: Ahmed H. Hanfy
"""

import cv2
import numpy as np
from .PerspectiveImageRebuild import CVColor
from linedrawingfunctions import IncParameters, IntersectionPoint

def cart2homo(leading_point: tuple[int, int], trailing_point: tuple[int, int], chord_vanish: list, 
                        vx: tuple[int, int], vy: tuple[int, int], 
                        curve_cart_points: list[tuple[float, float]], chord_length: float, angle_of_attack: float,
                        image: np.ndarray = None, aqu: int = 0, cir_radius: int = 1) -> list[tuple[int, int]]:
    
    """
    Calculate the homogeneous coordinate points of an curved surface (profile surface in case of aerofoil)
    from on given cartisian coordinates points refered to a baseline (Chord line in case of aerofoil) 
    and vanishing lines.

    :param leading_point: First point of the curved surface as a tuple (x, y).
    :param trailing_point: Last point of the curved surface as a tuple (x, y).
    .. note::
        - **First point of the curved surface** is the leaing edge point for the profile
        - **Last point of the curved surface** is the trailling edge point for the profile
        - Both first and last point of the curved surface should be laying on the chord line 
    :param chord_vanish: Vanishing point and chord angle in degrees as a list [vanish_point, chord_angle_in_deg].
    :param vx: Vanishing line x coordinates as a tuple (x, y).
    :param vy: Vanishing line y coordinates as a tuple (x, y).
    :param curve_cart_points: List of profile points as tuples of (x, y) coordinates.
    .. note::
        - The chord line represents the x-coordinates datum and the curve points are the y-coordinates
        - ``curve_cart_points`` points are shifted by the value of first point of the curved surface 
            (i.e. first point of the curved surface in cartisian coordintes y = 0 at x = 0 and the last
             point y = 0 at x = ...)
        - That dose not emply that the first point in ``curve_cart_points`` to be (0, 0)
    :param chord_length: Length of the chord.
    :param angle_of_attack: Angle of attack in degrees.
    :param aqu: Aqu value (default is 0).
    :param cir_radius: Circle radius for drawing (default is 1).
    :return: List of new profile points as tuples (x, y).

    :Example:

    >>> cart2homo((0, 0), (10, 0), [(5, 5), 45], (1, 1), (1, 1), [(0.5, 0.5)], 10, 5)
    [(5, 5)]

    .. note::
        currently "leading_point" and "trailing_point" are determine from the intersection between
        the chord lines and z-vanishing lines (both draw by the user) 

    """
    # Calculate the length of the chord in pixels.
    chord_pixel_length = np.linalg.norm(np.array(leading_point) - np.array(trailing_point))

    # Distance from the chord vanishing point to the leading edge chord point
    plp_chint = np.linalg.norm(np.array(chord_vanish[0]) - np.array(leading_point))

    # Distance from the chord vanishing point to the trailing edge chord point
    ptp_chint = np.linalg.norm(np.array(chord_vanish[0]) - np.array(trailing_point))

    new_profile_points = []
    for i, profile_point in enumerate(curve_cart_points):
        # Calculate the nominal point on the chord line
        # this divides the pixel cord length with the same raitos
        # of the provided points
        nom = chord_pixel_length * plp_chint * profile_point[0]
        pxdash_ptdash = chord_length - profile_point[0]

        # Nom = chord_pixel_length*PLPChint*ProfilePoints[i][0]
        # PxdashPTdash = CordLen - ProfilePoints[i][0]
        if ChordVanish[0][0] > TrailingChordPint[0]:
            LPx = Nom/(CordLen*PLPChint - PxdashPTdash*ChordInPixel)
        elif ChordVanish[0][0] < LeadingChordPint[0]:
            LPx = -Nom/(ProfilePoints[i][0]*ChordInPixel-PTPChint*CordLen)
        Pxx = round(LeadingChordPint[0]+LPx*np.cos(ChordVanish[1]*np.pi/180))
        Pxy = round(LeadingChordPint[1]+LPx*np.sin(ChordVanish[1]*np.pi/180))
        Px = (Pxx, Pxy)
        # Find Point distribution on Y-Vanishing lines
        myPx, ayPx = IncParameters(Vy, Px)
        mxPT, axPT = IncParameters(Vx, TrailingChordPint)

        VxPxint, VxPxAngle = IntersectionPoint([myPx, mxPT], [ayPx, axPT],
                                               [LeadingChordPint[0],
                                                LeadingChordPint[1]])

        PxPxVx = np.sqrt((Px[0]-VxPxint[0])**2+(Px[1]-VxPxint[1])**2)
        PxVy = np.sqrt((Px[0] - Vy[0])**2+(Px[1] - Vy[1])**2)
        PxdashPxVxdash = PxdashPTdash * np.sin(AOA)
        PpdashPxVxdash = ProfilePoints[i][1]+PxdashPxVxdash
        if PxdashPxVxdash*PxVy - PpdashPxVxdash*PxPxVx != 0:
            LPy = PxPxVx*PxVy*ProfilePoints[i][1]/(PxdashPxVxdash*PxVy-PpdashPxVxdash*PxPxVx)
            Ppx = round(LPy * np.cos(VxPxAngle[0] * np.pi / 180) + Pxx)
            if VxPxAngle[0] > 0:
                Ppy = round(Pxy - LPy * np.sin(np.deg2rad(VxPxAngle[0])))
            else:
                Ppy = round(Pxy - LPy * np.sin(-np.deg2rad(VxPxAngle[0])))
            Pp = (Ppx, Ppy)
            cv2.circle(clone, Pp, radius=cir_radius,
                       color=CVColor.BLUE, thickness=2)
            cv2.circle(clone, Pp, radius=cir_radius,
                       color=CVColor.RED, thickness=-1)
            cv2.putText(clone, f'{i+Aqu}', (Pp[0]-10, Pp[1]-10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
            NewProfilePoints.append(Pp)
        else:
            print('Unsupported point', ProfilePoints[i])
    return NewProfilePoints