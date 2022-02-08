#! /usr/bin/env python

from __future__ import absolute_import, division, print_function
from rospkg import RosPack
import sys
import os
# sys.path.append('../src/')
PATH = RosPack().get_path('depth_correction')
sys.path.append(os.path.join(PATH, 'src/'))
import torch
import numpy as np
from depth_correction.depth_cloud import DepthCloud
from depth_correction.loss import min_eigval_loss
from depth_correction.model import Linear, Polynomial
from depth_correction.filters import filter_depth, filter_grid, within_bounds
import rospy
from sensor_msgs.msg import PointCloud2
from ros_numpy import msgify
import tf
from torch.utils.tensorboard import SummaryWriter
import time


N_pts = 10000
r_nn = 0.4
LR = 0.001
N_ITERS = 200
min_depth = 1.0
max_depth = 15.0
grid_res = 0.05
PTS_HEIGHT = -10.0


def main():
    rospy.init_node('depth_correction', anonymous=True)
    pc_pub = rospy.Publisher('corrected_cloud', PointCloud2, queue_size=2)
    br = tf.TransformBroadcaster()

    # device = torch.device('cuda:0')
    device = torch.device('cpu')
    model = Polynomial(p0=(torch.rand(1,)*0.10+0.05),
                       p1=-(torch.rand(1,)*0.10+0.05), device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # create flat point cloud (a wall)
    cloud = np.zeros((N_pts, 3), dtype=np.float32)
    cloud[:, [0, 1]] = np.random.rand(N_pts, 2) * 20 - 10  # 10 x 10 m
    cloud[:, 2] = PTS_HEIGHT

    dc_init = DepthCloud.from_points(cloud)
    dc_init.to(device)
    dc_init = filter_depth(dc_init, min=min_depth, max=max_depth, log=False)
    dc_init = filter_grid(dc_init, grid_res, keep='last')
    dc_init.update_all(r=r_nn)
    # add disturbances
    p0, p1 = 0.1, -0.1
    dc_init.depth /= (1 - (p0 * dc_init.inc_angles ** 2 + p1 * dc_init.inc_angles ** 4))
    dc_init.update_all(r=r_nn)

    # dc_init.visualize(colors='min_eigval', normals=True)

    writer = SummaryWriter(os.path.join(PATH, 'scripts/tb_runs/model_Polynomial_lr_%f' % LR))
    # use model to correct the distortion (hopefully)
    for i in range(N_ITERS):
        if rospy.is_shutdown():
            break
        t0 = time.time()
        optimizer.zero_grad()

        dc = model(dc_init)

        loss, _ = min_eigval_loss(dc, r=r_nn, offset=True, eigenvalue_bounds=(0.0, 0.05 ** 2))

        rospy.loginfo('Loss: %g' % loss.item())

        # Optimization step
        loss.backward()
        optimizer.step()

        dc.update_all(r=r_nn)
        rospy.loginfo('Optimization step took %.3f sec' % (time.time()-t0))

        # Tensorboard logging
        writer.add_scalar("Loss/min_eigval", loss, i)
        writer.add_scalars('ModelParams0', {'p_0': model.p0, 'p_0_distortion': p0}, i)
        writer.add_scalars('ModelParams1', {'p_1': model.p1, 'p_1_distortion': p1}, i)
        eps = 0.02
        depth_keep = within_bounds(dc.points[..., 2], PTS_HEIGHT-eps, PTS_HEIGHT+eps, log_variable='Z')
        writer.add_scalars("FlatPointsRatio", {'eps %f' % eps: depth_keep.double().mean()}, i)

        if i % 10 == 0:
            # dc.visualize(colors='inc_angles', normals=True)

            # publish point cloud msg
            n_pts = dc.points.shape[0]
            cloud = np.zeros((n_pts,), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                              ('eig_x', 'f4'), ('eig_y', 'f4'), ('eig_z', 'f4')])
            points = dc.points.detach().cpu().numpy()
            eigs = dc.eigvals.detach().cpu().numpy()
            cloud['x'], cloud['y'], cloud['z'] = points[:, 0], points[:, 1], points[:, 2]
            cloud['eig_x'], cloud['eig_y'], cloud['eig_z'] = eigs[:, 0], eigs[:, 1], eigs[:, 2]
            # cloud = DepthCloud.to_structured_array(dc)
            pc_msg = msgify(PointCloud2, cloud)
            pc_msg.header.frame_id = 'map'
            pc_msg.header.stamp = rospy.Time.now()
            pc_pub.publish(pc_msg)

            # publish viewpoints
            vp = dc.vps[0]
            br.sendTransform((vp[0], vp[1], vp[2]), (0, 0, 0, 1), rospy.Time.now(), "vp", "map")


if __name__ == '__main__':
    main()
