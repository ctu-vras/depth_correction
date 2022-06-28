import torch
from typing import Union
from pytorch3d.ops.knn import knn_points
from pytorch3d.structures import Pointclouds


def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.
    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
            lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths


def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    apply_point_reduction=True,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.
    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        apply_point_reduction: Whether to apply points reduction. If set to True returns
            one distance per batch (of shape (N, 1)). Otherwise, result size is equal to (N, P1).
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    Returns:
        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
    """
    x, x_lengths = _handle_pointcloud_input(x, x_lengths)
    y, y_lengths = _handle_pointcloud_input(y, y_lengths)

    N, P1, D = x.shape

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)

    # knn_points returns squared distances
    cham_x = x_nn.dists[..., 0].sqrt()  # (N, P1)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    if apply_point_reduction:
        # Apply point reduction
        cham_x = cham_x.sum(1)  # (N,)

        if point_reduction == "mean":
            cham_x /= x_lengths

        if batch_reduction is not None:
            # batch_reduction == "sum"
            cham_x = cham_x.sum()

            if batch_reduction == "mean":
                cham_x /= N

    return cham_x
