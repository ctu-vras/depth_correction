#! /usr/bin/env python

from __future__ import absolute_import, division, print_function
from rospkg import RosPack
import sys
import os
sys.path.append(os.path.join(RosPack().get_path('depth_correction'), 'src/'))
import torch
import numpy as np
from depth_correction.depth_cloud import DepthCloud
from depth_correction.loss import min_eigval_loss
from depth_correction.model import Linear, Polynomial
import rospy
from sensor_msgs.msg import PointCloud2
from ros_numpy import msgify, numpify


N = 10000
r_nn = 0.4
LR = 0.001


def main():
    rospy.init_node('depth_correction', anonymous=True)
    pc_pub = rospy.Publisher('corrected_cloud', PointCloud2, queue_size=2)

    device = torch.device('cpu')
    model = Polynomial(p0=0.05, p1=-0.05, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # create flat point cloud (a wall)
    cloud = np.zeros((N, 3), dtype=np.float32)
    cloud[:, :2] = np.random.rand(N, 2) * 20 - 10  # 10 x 10 m
    cloud[:, 2] = -10.0

    dc = DepthCloud.from_points(cloud)
    dc.update_all(r=r_nn)

    # add disturbance: modify Z coord of point with high incidence angle
    # alpha = dc.inc_angles.squeeze()
    # min_alpha, max_alpha = 0.4, 0.7
    # keep = alpha < max_alpha
    # keep = keep & keep > min_alpha
    # dc.depth[keep] = dc.depth[keep] + 0.3

    # dc.depth = dc.depth + torch.exp(-torch.sin((dc.inc_angles**2) / 0.3 ** 2))
    dc.update_all(r=r_nn)

    # dc.visualize(colors='min_eigval')

    # use model to correct the distortion (hopefully)
    for i in range(200):
        if rospy.is_shutdown():
            break
        optimizer.zero_grad()

        dc = model(dc)
        # dc.update_all(r=r_nn)

        loss, _ = min_eigval_loss(dc, r=r_nn, offset=True)

        rospy.loginfo('loss: %f' % loss.item())

        # Optimization step
        loss.backward(retain_graph=True)
        optimizer.step()

        # if i % 20 == 0:
        #     # dc.visualize(colors='inc_angles')
        #     dc.visualize(colors='min_eigval')

        # publish point cloud msg
        dc.update_points()
        n_pts = dc.points.shape[0]
        cloud = np.zeros((n_pts,), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        points = dc.points.detach().numpy()
        cloud['x'], cloud['y'], cloud['z'] = points[:, 0], points[:, 1], points[:, 2]
        pc_msg = msgify(PointCloud2, cloud)
        pc_msg.header.frame_id = 'map'
        pc_msg.header.stamp = rospy.Time.now()
        pc_pub.publish(pc_msg)


if __name__ == '__main__':
    main()
