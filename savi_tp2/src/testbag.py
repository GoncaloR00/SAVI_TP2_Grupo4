#!/usr/bin/env python3

import open3d as o3d
bag_reader = o3d.t.io.RSBagReader()
bag_reader.open('./kinect_complete.bag')
im_rgbd = bag_reader.next_frame()
while not bag_reader.is_eof():
    # process im_rgbd.depth and im_rgbd.color
    im_rgbd = bag_reader.next_frame()

bag_reader.close()