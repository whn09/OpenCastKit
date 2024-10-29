import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
# from ffrecord import FileReader
# from ffrecord.torch import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
import data_factory.graph_tools as gg

import os
import time
import random
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

import torch
from torch.utils import data


class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def load(self, scaler_dir):
        with open(scaler_dir, "rb") as f:
            pkl = pickle.load(f)
            self.mean = pkl["mean"]
            self.std = pkl["std"]

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


class ERA5(Dataset):

    def __init__(self, split: str, check_data: bool = True, modelname: str = 'fourcastnet') -> None:

        self.data_dir = Path("./output/data/")

        assert split in ["train", "val"]
        assert modelname in ["fourcastnet", "graphcast"]
        self.split = split
        self.modelname = modelname
        self.fname = str(self.data_dir / f"{split}.ffr")
        # self.reader = FileReader(self.fname, check_data)
        self.reader = open(self.fname, 'r')
        self.scaler = StandardScaler()
        self.scaler.load("./output/data/scaler.pkl")

        if self.modelname == 'graphcast':
            self.constant_features = gg.fetch_constant_features()
        else:
            self.constant_features = None

    def __len__(self):
        return self.reader.n

    def __getitem__(self, indices):
        seqs_bytes = self.reader.read(indices)
        samples = []
        for bytes_ in seqs_bytes:
            x0, x1, y = pickle.loads(bytes_)

            if self.modelname == 'fourcastnet':
                x0 = np.nan_to_num(x0[:, :, :-2])
                x1 = np.nan_to_num(x1[:, :, :-2])
                y = np.nan_to_num(y[:, :, :-2])
                samples.append((x0, x1, y))
            else:
                x = np.nan_to_num(np.reshape(np.concatenate([x0, x1, y[:, :, -2:]], axis=-1), [-1, 49]))
                y = np.nan_to_num(np.reshape(y[:, :, :-2], [-1, 20]))
                samples.append((x, y))
        return samples

    def get_scaler(self):
        return self.scaler

    def loader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self, *args, **kwargs)

class PTDataset(data.Dataset):
    """Dataset class for the era5 upper and surface variables."""

    def __init__(self,
                 pt_path='/fsx/datalab/nsf-ncar-era5',
                 data_transform=None,
                 seed=1234,
                 training=True,
                 validation=False,
                 startDate='20150101 12:00:00',
                 endDate='20150201 12:00:00',
                 freq='24H',
                 horizon=24,
                 device='cpu'):
        """Initialize."""
        self.horizon = horizon
        self.device = device
        self.pt_path = pt_path
        """
        To do
        if start and end is valid date, if the date can be found in the downloaded files, length >= 0

        """
        # Prepare the datetime objects for training, validation, and test
        self.training = training
        self.validation = validation
        self.data_transform = data_transform

        if training:
            self.keys = list(pd.date_range(
                start=startDate, end=endDate, freq=freq))
            # self.keys = (list(set(self.keys))) #disordered keys
            # total length that we can predict
            """
            To do
            length should >=0 horizon <= len
            """
        elif validation:
            self.keys = list(pd.date_range(
                start=startDate, end=endDate, freq=freq))
            # self.keys = (list(set(self.keys)))

        else:
            self.keys = list(pd.date_range(
                start=startDate, end=endDate, freq=freq))
            # self.keys = (list(set(self.keys)))
            # end_time = self.keys[0] + timedelta(hours = self.horizon)
        # self.length = len(self.keys) - horizon // 12 - 1  # TODO Why horizon // 12 ?
        self.length = len(self.keys) - horizon // int(freq[:-1]) - 1

        print('self.keys:', len(self.keys), self.keys[:10], self.keys[-10:])
        print('self.length:', self.length)

        random.seed(seed)

    def LoadData(self, key):
        """
        Input
            key: datetime object, input time
        Return
            input: numpy
            input_surface: numpy
            target: numpy label
            target_surface: numpy label
            (start_time_str, end_time_str): string, datetime(target time - input time) = horizon
        """
        # start_time datetime obj
        start_time = key
        # convert datetime obj to string for matching file name and return key
        start_time_str = datetime.strftime(key, '%Y%m%d%H')

        # target time = start time + horizon
        # TODO: 这里有问题，其实是把horizon当timestep用了
        end_time = key + timedelta(hours=self.horizon)
        end_time_str = end_time.strftime('%Y%m%d%H')

        # print('start_time_str:', start_time_str)
        # print('end_time_str:', end_time_str)

        # device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
        # device = 'cpu'
        # print('device:', device)

        # Prepare the input_surface dataset
        # print(start_time_str[0:6])
        input_surface = torch.load(os.path.join(self.pt_path, 'surface', 'surface_{}.pt'.format(
            start_time_str)), weights_only=False, map_location=self.device)  # 201501

        # Prepare the input_upper dataset
        input = torch.load(os.path.join(self.pt_path, 'upper', 'upper_{}.pt'.format(
            start_time_str)), weights_only=False, map_location=self.device)

        # print('input:', input.shape)
        # print('input_surface:', input_surface.shape)
        assert input_surface.shape == (4, 721, 1440)
        assert input.shape == (5, 13, 721, 1440)

        # Prepare the target_surface dataset
        target_surface = torch.load(os.path.join(self.pt_path, 'surface', 'surface_{}.pt'.format(
            end_time_str)), weights_only=False, map_location=self.device)  # 201501

        # Prepare the target upper dataset
        target = torch.load(os.path.join(self.pt_path, 'upper', 'upper_{}.pt'.format(
            end_time_str)), weights_only=False, map_location=self.device)

        # print('target:', target.shape)
        # print('target_surface:', target_surface.shape)
        assert target_surface.shape == (4, 721, 1440)
        assert target.shape == (5, 13, 721, 1440)

        return input, input_surface, target, target_surface, (start_time_str, end_time_str)

    def __getitem__(self, index):
        """Return input frames, target frames, and its corresponding time steps."""

        iii = self.keys[index]
        LoadData_start = time.time()
        input, input_surface, target, target_surface, periods = self.LoadData(
            iii)
        LoadData_end = time.time()
        # print('LoadData time:', LoadData_end-LoadData_start)

        if self.training:
            if self.data_transform is not None:
                input = self.data_transform(input)
                input_surface = self.data_transform(input_surface)

        return input, input_surface, target, target_surface, periods

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__

class EarthGraph(object):
    def __init__(self):
        self.mesh_data = None
        self.grid2mesh_data = None
        self.mesh2grid_data = None

    def generate_graph(self):
        mesh_nodes = gg.fetch_mesh_nodes()

        mesh_6_edges, mesh_6_edges_attrs = gg.fetch_mesh_edges(6)
        mesh_5_edges, mesh_5_edges_attrs = gg.fetch_mesh_edges(5)
        mesh_4_edges, mesh_4_edges_attrs = gg.fetch_mesh_edges(4)
        mesh_3_edges, mesh_3_edges_attrs = gg.fetch_mesh_edges(3)
        mesh_2_edges, mesh_2_edges_attrs = gg.fetch_mesh_edges(2)
        mesh_1_edges, mesh_1_edges_attrs = gg.fetch_mesh_edges(1)
        mesh_0_edges, mesh_0_edges_attrs = gg.fetch_mesh_edges(0)

        mesh_edges = mesh_6_edges + mesh_5_edges + mesh_4_edges + mesh_3_edges + mesh_2_edges + mesh_1_edges + mesh_0_edges
        mesh_edges_attrs = mesh_6_edges_attrs + mesh_5_edges_attrs + mesh_4_edges_attrs + mesh_3_edges_attrs + mesh_2_edges_attrs + mesh_1_edges_attrs + mesh_0_edges_attrs

        self.mesh_data = Data(x=torch.tensor(mesh_nodes, dtype=torch.float),
                              edge_index=torch.tensor(mesh_edges, dtype=torch.long).T.contiguous(),
                              edge_attr=torch.tensor(mesh_edges_attrs, dtype=torch.float))

        grid2mesh_edges, grid2mesh_edge_attrs = gg.fetch_grid2mesh_edges()
        self.grid2mesh_data = Data(x=None,
                                   edge_index=torch.tensor(grid2mesh_edges, dtype=torch.long).T.contiguous(),
                                   edge_attr=torch.tensor(grid2mesh_edge_attrs, dtype=torch.float))

        mesh2grid_edges, mesh2grid_edge_attrs = gg.fetch_mesh2grid_edges()
        self.mesh2grid_data = Data(x=None,
                                   edge_index=torch.tensor(mesh2grid_edges, dtype=torch.long).T.contiguous(),
                                   edge_attr=torch.tensor(mesh2grid_edge_attrs, dtype=torch.float))