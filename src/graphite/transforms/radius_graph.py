from torch_geometric.transforms import BaseTransform
from torch_geometric.nn import radius_graph


class RadiusGraph(BaseTransform):
    r"""Creates edges based on node positions :obj:`pos` to all points within a
    given distance (functional name: :obj:`radius_graph`).

    This transform class is a slighly modified version of PyG's RadiusGraph.
    This version additionally calculates edge vectors as `edge_attr`.

    Args:
        r (float): The distance.
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`.
            This flag is only needed for CUDA tensors. (default: :obj:`32`)
        flow (string, optional): The flow direction when using in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
    """
    def __init__(
        self,
        r: float,
        loop: bool = False,
        max_num_neighbors: int = 32,
        flow: str = 'source_to_target',
        num_workers: int = 1,
    ):
        self.r = r
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors
        self.flow = flow
        self.num_workers = num_workers

    def __call__(self, data):
        batch = data.batch if 'batch' in data else None

        data.edge_index = radius_graph(
            data.pos,
            self.r,
            batch,
            self.loop,
            max_num_neighbors=self.max_num_neighbors,
            flow=self.flow,
            num_workers=self.num_workers,
        )

        i, j = data.edge_index
        data.edge_attr = data.pos[j] - data.pos[i]

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(r={self.r})'