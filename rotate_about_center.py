#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 14:50:51 2018

@author: duli
"""
import numpy as np

import cv2

import math


def rotate_about_center(src, angle, scale=1.):
  w = src.shape[1]
  h = src.shape[0]
#  rangle = np.deg2rad(angle) # angle in radians
  # now calculate new image width and height
  nw = (abs(np.sin(angle)*h) + abs(np.cos(angle)*w))*scale
  nh = (abs(np.cos(angle)*h) + abs(np.sin(angle)*w))*scale
  # ask OpenCV for the rotation matrix
  rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
  # calculate the move from the old center to the new center combined
  # with the rotation
  rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
  # the move only affects the translation, so update the translation
  # part of the transform
  rot_mat[0,2] += rot_move[0]
  rot_mat[1,2] += rot_move[1]
  return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
