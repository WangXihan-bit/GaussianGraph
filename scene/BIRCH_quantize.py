import sys
sys.path.append('/home/wangxihan/OpenGaussian/')
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from open3d import visualization
from sklearn.cluster import DBSCAN
from plyfile import PlyData
from utils.birch_utils import Birch
from gaussian_renderer import GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

class Cf_cluster():

    def __init__(self, feat_scale=1, voxel_size=0.02, radius=0.1, max_nn=50, control_points_num=5000, branching_factor=100):
        self.feat_scale = feat_scale
        self.voxel_size = voxel_size
        self.radius = radius
        self.max_nn = max_nn
        self.control_points_num = control_points_num
        self.branching_factor = branching_factor
        self.centers = torch.empty(0)
        self.cls_ids = torch.empty(0)
        self.follow_cls_ids = torch.empty(0)

    def control_points(self, gaussians):
        # 采样稀疏控制点需要考虑语义梯度和空间位置，目前只包括空间位置
        # 需要对比最远点采样和FPFS采样
        #_, self.denoise_indices = self.denoise(gaussians._ins_feat)
        #sample_points, sample_indices = self.fpfh_sampling(gaussians._xyz[self.denoise_indices].cpu().numpy())
        self.device = gaussians._xyz.device
        sample_points, sample_indices = self.fpfh_sampling(gaussians._xyz.detach().cpu().numpy())
        self.sample_points = torch.from_numpy(np.asarray(sample_points.points)).to(self.device)
        sample_indices = torch.from_numpy(sample_indices).to(self.device)
        #self.control_indices = self.denoise_indices[sample_indices]
        self.control_indices = sample_indices
        self.vec_dim = 9
        self.sample_features = gaussians._ins_feat[self.control_indices].to(self.device)

        self.all_cluster_ids = torch.zeros(gaussians._xyz.shape[0], dtype=torch.long, device=self.device)
        all_indices = torch.arange(gaussians._xyz.shape[0], device=self.device)
        self.remain_indices = all_indices[~torch.isin(all_indices, self.control_indices)]
        self.remain_points = gaussians._xyz[self.remain_indices].to(self.device)
        self.remain_features = gaussians._ins_feat[self.remain_indices].to(self.device)

    def cluster_group(self, mode, iteration, opt):

        if mode == "control":
            if iteration == opt.start_control_cb_iter + 1:
                self.partial = False
                feat_std = self.sample_features.std(dim=0).mean().item()  # 取所有特征的平均标准差
                xyz_std = self.sample_points.std(dim=0).mean().item()  # 取空间坐标的标准差平均值
                threshold = (feat_std  + xyz_std) / 2
                self.birch_model = Birch(threshold=threshold, branching_factor=self.branching_factor, n_clusters=None)
            else:
                self.partial = True
            cluster_ids, cluster_centers, birch_model = self.birch(self.sample_features, self.sample_points, self.partial, self.birch_model)
            self.all_cluster_ids[self.control_indices] = cluster_ids + 1
            self.control_cls_ids = self.all_cluster_ids
            self.control_centers = cluster_centers
            self.cluster_num = max(cluster_ids) + 1
            self.birch_model = birch_model
            
        # 可视化采样点BIRCH聚类结果
        #self.visualize_clusters(sample_points, cluster_ids)

        elif mode == "follow":
            # BIRCH.predict是否可以替代match_remain_points???
            if iteration == opt.start_follow_cb_iter + 1:
                self.cluster_centers = self.control_centers
            self.cluster_centers, cluster_num, remain_cluster_ids = self.match_remain_points(torch.cat([self.remain_features*self.feat_scale, self.remain_points],dim=1), self.cluster_centers)
            self.all_cluster_ids[self.remain_indices] = remain_cluster_ids
            self.follow_cls_ids = self.all_cluster_ids
            self.follow_centers = self.cluster_centers    
            #self.visualize_clusters(gaussians._xyz, all_cluster_ids)
            self.cluster_num = cluster_num

    def denoise(self, features):
        # 将PyTorch张量转换为NumPy数组供DBSCAN使用
        features_np = features.cpu().numpy()

        # 使用DBSCAN进行聚类
        db = DBSCAN(eps=0.5, min_samples=10)
        labels = db.fit_predict(features_np)

        # 在标签中，-1表示噪声点，我们需要过滤掉这些噪声点
        noise_points_mask = labels == -1

        # 提取去噪后的数据（去除噪声点）
        clean_points = features[~torch.tensor(noise_points_mask).cuda()]
        # 获取去噪后数据的索引
        clean_indices = torch.nonzero(~torch.tensor(noise_points_mask).cuda()).squeeze()

        return clean_points, clean_indices

    def fpfh_sampling(self, points):
        """
        从给定的点云中进行最远点采样并返回采样的点以及其对应的索引。

        Args:
            points (numpy.ndarray): 输入的点云数据，形状为 (N, 3)。
            num_samples (int): 需要采样的点的数量。

        Returns:
            sampled_points (numpy.ndarray): 采样得到的点，形状为 (num_samples, 3)。
            indices (list): 被采样点的索引列表。
        """
        # 将输入的点云转为 open3d 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 下采样点云
        voxel_down_pcd = pcd.voxel_down_sample(self.voxel_size)

        # 计算法线
        voxel_down_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(self.radius, self.max_nn))

        # 计算FPFH特征
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            voxel_down_pcd,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(self.radius, self.max_nn))

        # 获取关键点索引
        keypoints_indices = np.argpartition(np.asarray(fpfh.data).sum(axis=0), -self.control_points_num)[-self.control_points_num:]
        keypoints_cloud = voxel_down_pcd.select_by_index(keypoints_indices)

        return keypoints_cloud, keypoints_indices

    def birch(self, feat, xyz, partial, birch_model):

        cluster_feature = torch.cat([feat*self.feat_scale, xyz], dim=1)
        birch_model.fit(cluster_feature, partial)

        # Get cluster assignments for control clusters
        cls_ids = torch.tensor(birch_model.predict(cluster_feature), dtype=torch.long).to(self.device) # [num_pts]
        # Extract control cluster centers (centroids)
        centers = torch.tensor(birch_model.subcluster_centers_, dtype=torch.float32).to(self.device) # [k1, 9]

        return cls_ids, centers, birch_model

    def get_dist(self, features, cluster_feats, mode='sq_euclidean_chunk'):
        """
        计算给定特征与聚类中心之间的距离。
        
        features: 当前待处理的特征 (batch_size, feature_dim)
        cluster_feats: 已知的聚类中心特征 (cluster_num, feature_dim)
        mode: 距离计算方式 ('sq_euclidean_chunk' 或 'cosine')
        
        返回:
        dist: 特征与聚类中心之间的距离 (batch_size, cluster_num)
        """
        if mode == 'sq_euclidean_chunk':
            # 计算欧几里得距离（平方）
            dist = torch.cdist(features, cluster_feats, p=2) ** 2
        elif mode == 'cosine':
            # 计算余弦距离
            features_norm = F.normalize(features, p=2, dim=-1)
            cluster_feats_norm = F.normalize(cluster_feats, p=2, dim=-1)
            dist = 1 - torch.mm(features_norm, cluster_feats_norm.T)
        return dist

    def update_centers_(self, old_features, new_features, dist, curr_nn_index, cluster_weight = 0.9):
        """
        更新聚类中心。
        
        features: 当前聚类中的特征 (batch_size, feature_dim)
        dist: 特征与聚类中心的归类情况 (batch_size, cluster_num)
        curr_nn_index: 当前聚类中心索引
        avg: 是否使用平均值更新聚类中心
        
        返回:
        updated_centers: 更新后的聚类中心 (cluster_num, feature_dim)
        """
        updated_centers = []
        cluster_num = dist.shape[1]  # 聚类中心的数量
        for i in range(cluster_num):
            # 获取当前聚类中所有归类到该聚类的特征
            new_cluster_features = new_features[dist[:, i] == 1]
            if new_cluster_features.size(0) > 0:
                
                # 计算合并后的平均值作为新的聚类中心
                updated_center = cluster_weight * old_features[i] + (1 - cluster_weight) * new_cluster_features.mean(dim=0)
                updated_centers.append(updated_center)
            else:
                # 如果当前聚类没有任何点，保留原聚类中心
                updated_centers.append(old_features[i])

        return torch.stack(updated_centers)

    def match_remain_points(self, remain_feats, cluster_feats, threshold=0.1, chunk=10000):
        """
        对特征点进行聚类匹配，更新聚类中心，形成新的聚类。
        
        remain_feats: 所有剩余跟随点的特征 [N, feature_dim]
        cluster_feats: 已知的聚类中心特征 [cluster_num, feature_dim]
        threshold: 距离阈值，小于该阈值则归类为已有聚类，否则形成新聚类
        chunk: 分块大小，减少内存占用
        
        返回:
        - updated_feats: 更新后的特征
        - updated_cluster_feats: 更新后的聚类中心
        - updated_cluster_num: 更新后的聚类数量
        """
        updated_feats = remain_feats
        updated_cluster_feats = cluster_feats
        updated_cluster_num = cluster_feats.shape[0]
        remain_num = remain_feats.size(0)  
        cluster_ids = torch.zeros(remain_num, dtype=torch.long).to(remain_feats.device)

        for i in range(0, remain_num, chunk):
            # 处理每个块
            end_idx = min(i + chunk, remain_num)
            chunk_feats = remain_feats[i:end_idx]

            # 计算该块特征与聚类中心的距离
            dist = self.get_dist(chunk_feats, updated_cluster_feats, mode='sq_euclidean_chunk')
            curr_nn_index = torch.argmin(dist, dim=-1) + 1  # shape: [chunk_num, cluster_num]
            # 是否转成one-hot编码？？？

            # 归一化距离到 [0, 1]
            dist_min = dist.min()
            dist_max = dist.max()
            normalized_dist = (dist - dist_min) / (dist_max - dist_min)  

            # 判断是否符合归类条件
            # 将距离小于阈值的点归为已有聚类
            cluster_ids[i:end_idx][normalized_dist.min(dim=-1).values <= threshold] = curr_nn_index[normalized_dist.min(dim=-1).values < threshold]
            # 将距离大于阈值的点归为新聚类
            new_feats_mask = normalized_dist.min(dim=-1).values > threshold
            new_feats = chunk_feats[new_feats_mask]

            # 检测新聚类能否合并
            if new_feats.size(0) > 0:
                # 创建新的聚类（每个点分配一个类别）
                new_cluster_ids = torch.arange(updated_cluster_num + 1, updated_cluster_num + new_feats.size(0) + 1, device=remain_feats.device)
                cluster_ids[i:end_idx][new_feats_mask] = new_cluster_ids  # 给新点分配新的聚类 ID

                updated_cluster_feats = torch.cat([updated_cluster_feats, new_feats], dim=0)

                # 计算新聚类之间的距离矩阵
                new_dist = self.get_dist(new_feats, new_feats, mode='sq_euclidean_chunk')
        
                # 归一化距离到 [0, 1]
                dist_min = new_dist.min()
                dist_max = new_dist.max()
                normalized_dist = (new_dist - dist_min) / (dist_max - dist_min)

                # 创建合并条件：新聚类之间的距离小于阈值，表示可以合并
                merge_mask = normalized_dist <= threshold
                # 标记是否被删除的聚类中心
                deleted = torch.zeros(updated_cluster_feats.shape[0], dtype=torch.bool)
                visited = torch.zeros([updated_cluster_feats.shape[0], updated_cluster_feats.shape[0]], dtype=torch.bool)

                # 如果合并条件成立，合并聚类
                if merge_mask.any():
                    # 计算合并后的聚类 ID
                    merge_indices = torch.where(merge_mask)  # 找到需要合并的聚类对
                    
                    for idx1, idx2 in zip(*merge_indices):
                        if idx1 != idx2:  # 防止自己合并自己
                            idx1, idx2 = idx1+cluster_feats.shape[0]+1, idx2+cluster_feats.shape[0]+1

                            # 确保每对聚类中心只合并一次
                            if visited[idx1][idx2] or visited[idx2][idx1]:
                                continue

                            # 将属于 idx2 的聚类点合并到 idx1
                            cluster_ids[(cluster_ids == idx2) | (cluster_ids == idx1)] = min(idx1, idx2)

                            # 更新聚类中心为两者的均值
                            updated_cluster_feats[min(idx1, idx2)] = (updated_cluster_feats[idx1] * len(cluster_ids[cluster_ids == idx1]) 
                                                        + updated_cluster_feats[idx2] * len(cluster_ids[cluster_ids == idx2]) 
                                                        ) / (len(cluster_ids[cluster_ids == idx1]) + len(cluster_ids[cluster_ids == idx2]))

                            # 删除 idx2 聚类中心
                            deleted[max(idx1,idx2)] = True
                            visited[idx1][idx2] =True
                            visited[idx2][idx1] =True

                updated_cluster_feats = updated_cluster_feats[~deleted]

        updated_cluster_num = updated_cluster_feats.shape[0]

        return updated_cluster_feats, updated_cluster_num, cluster_ids

    def sigmoid(x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def visualize_ply(self, ply_path):
        # Load the PLY file
        ply_data = PlyData.read(ply_path)
        vertex_data = ply_data['vertex'].data

        # Extract the point cloud attributes
        points = np.array([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
        colors = np.array([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T / 255.0
        opacity = vertex_data['opacity']

        # Apply the opacity filter
        sigmoid_opacity = self.sigmoid(opacity)
        filtered_indices = sigmoid_opacity >= 0.1
        filtered_points = points[filtered_indices]
        filtered_colors = colors[filtered_indices]

        # Create an Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])

    def visualize_clusters(xyz, cls_ids):
        """
        可视化聚类结果
        :param xyz: 点云坐标, shape: [num_pts, 3]
        :param cls_ids: 点云的聚类标签, shape: [num_pts]
        :param centers: 聚类中心, shape: [num_clusters, 3]
        """
        # 将数据转换为 open3d 点云对象
        xyz_np = xyz.cpu().numpy()  # 转换为 NumPy 数组
        cls_ids = cls_ids.cpu().numpy()

        # 获取最大类别数并创建颜色映射
        unique_cls_ids = np.unique(cls_ids)
        n_cls = len(unique_cls_ids)
        
        # 为每个类别生成随机颜色 (RGB值在 [0, 1] 之间)
        np.random.seed(42)  # 为了确保每次生成相同的随机颜色
        random_colors = np.random.rand(n_cls, 3)  # 生成随机 RGB 颜色，形状是 [n_cls, 3]

        # 为每个点分配颜色
        colors = np.array([random_colors[np.where(unique_cls_ids == cls_id)[0][0]] for cls_id in cls_ids])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_np)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])

    
    def forward(self, gaussian, mode, iteration, opt):
        
        # 每轮重新选择控制点还是使用相同的控制点？？？
        # self.control_points(gaussians=gaussian)
        self.cluster_group(mode, iteration, opt)
    
        if mode == "control":
            centers = self.control_centers
            self.nn_index = self.control_cls_ids - 1
        elif mode == "follow":
            centers = self.follow_centers
            self.nn_index = self.follow_cls_ids - 1
        
        # 检查 nn_index 是否超出范围
        out_of_bounds_mask = (self.nn_index < -1) | (self.nn_index > centers.shape[0])

        if torch.any(out_of_bounds_mask):
            # 打印出超出范围的索引值
            out_of_bounds_indices = self.nn_index[out_of_bounds_mask]
            print("Out of bounds indices:", out_of_bounds_indices)
    
            # 抛出 AssertionError
            raise AssertionError(f"nn_index is out of the bound of centers.")
        
        if mode == "control":
            valid_mask = self.nn_index != -1
            sampled_centers = torch.zeros((self.nn_index.shape[0], self.vec_dim), device=self.device)
            sampled_centers[valid_mask] = torch.gather(centers, 0, self.nn_index[valid_mask].unsqueeze(-1).repeat(1, self.vec_dim))
        elif mode == "follow":
            sampled_centers = torch.gather(centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        
        # NOTE: "During backpropagation, the gradients of the quantized features are copied to the instance features", mentioned in the paper.
        # _ins_feat_q 在数值上等于 sampled_centers[:, :6]，但在反向传播时，梯度会直接传递给原始的 _ins_feat
        gaussian._ins_feat_q = gaussian._ins_feat - gaussian._ins_feat.detach() + sampled_centers[:,:6]
        # 在反向传播时，确保只对有效索引计算梯度，其他部分的梯度不会传播
        # gaussian._ins_feat_q = gaussian._ins_feat_q * valid_mask.unsqueeze(-1).float()  # 只对有效索引进行梯度计算



