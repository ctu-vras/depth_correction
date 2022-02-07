#! /usr/bin/env python

from __future__ import absolute_import, division, print_function
from rospkg import RosPack
import sys
import os
# sys.path.append('../src/')
sys.path.append(os.path.join(RosPack().get_path('depth_correction'), 'src/'))
import torch
import numpy as np
from depth_correction.depth_cloud import DepthCloud
from depth_correction.loss import min_eigval_loss
from depth_correction.model import Linear, Polynomial
from depth_correction.filters import filter_depth, filter_grid
import rospy
from sensor_msgs.msg import PointCloud2
from ros_numpy import msgify
import tf


N_pts = 10000
r_nn = 0.4
LR = 1e-2
N_ITERS = 200
SHOW_RESULTS = True
min_depth = 1.0
max_depth = 15.0
grid_res = 0.05


# define normalized 2D gaussian: https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1, normalize=True):
    gauss = np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))
    if normalize:
        gauss /= (2. * np.pi * sx * sy)
    return gauss


def main():
    rospy.init_node('depth_correction', anonymous=True)
    pc_pub = rospy.Publisher('corrected_cloud', PointCloud2, queue_size=2)
    br = tf.TransformBroadcaster()

    device = torch.device('cpu')
    model = Polynomial(p0=1e-3, p1=1e-3, device=device)
    # model = Linear(w0=1.0 + 1e-3, w1=0.0, b=1e-3, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # create flat point cloud (a wall)
    cloud = np.zeros((N_pts, 3), dtype=np.float32)
    cloud[:, [0, 1]] = np.random.rand(N_pts, 2) * 20 - 10  # 10 x 10 m
    cloud[:, 2] = -10.0
    # add disturbances
    cloud[:, 2] += gaus2d(cloud[:, 0], cloud[:, 1], mx=6, my=6, normalize=False)
    cloud[:, 2] += gaus2d(cloud[:, 0], cloud[:, 1], mx=6, my=-6, normalize=False)
    cloud[:, 2] += gaus2d(cloud[:, 0], cloud[:, 1], mx=-6, my=6, normalize=False)
    cloud[:, 2] += gaus2d(cloud[:, 0], cloud[:, 1], mx=-6, my=-6, normalize=False)

    dc_init = DepthCloud.from_points(cloud)
    dc_init = filter_depth(dc_init, min=min_depth, max=max_depth, log=False)
    dc_init = filter_grid(dc_init, grid_res, keep='last')
    dc_init.update_all(r=r_nn)

    # dc_init.visualize(colors='min_eigval', normals=True)

    # use model to correct the distortion (hopefully)
    for i in range(N_ITERS):
        if rospy.is_shutdown():
            break
        optimizer.zero_grad()

        dc = model(dc_init)

        loss, _ = min_eigval_loss(dc, r=r_nn, offset=True, eigenvalue_bounds=(0.0, 0.05 ** 2))

        rospy.loginfo('Loss: %g' % loss.item())

        # Optimization step
        loss.backward()
        optimizer.step()

        dc.update_all(r=r_nn)

        if i % 20 == 0 and SHOW_RESULTS:
            dc.visualize(colors='inc_angles', normals=True)

        # publish point cloud msg
        n_pts = dc.points.shape[0]
        cloud = np.zeros((n_pts,), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                          ('eig_x', 'f4'), ('eig_y', 'f4'), ('eig_z', 'f4')])
        points = dc.points.detach().numpy()
        eigs = dc.eigvals.detach().numpy()
        cloud['x'], cloud['y'], cloud['z'] = points[:, 0], points[:, 1], points[:, 2]
        cloud['eig_x'], cloud['eig_y'], cloud['eig_z'] = eigs[:, 0], eigs[:, 1], eigs[:, 2]
        pc_msg = msgify(PointCloud2, cloud)
        pc_msg.header.frame_id = 'map'
        pc_msg.header.stamp = rospy.Time.now()
        pc_pub.publish(pc_msg)

        # publish viewpoints
        vp = dc.vps[0]
        br.sendTransform((vp[0], vp[1], vp[2]), (0, 0, 0, 1), rospy.Time.now(), "vp", "map")


if __name__ == '__main__':
    main()
