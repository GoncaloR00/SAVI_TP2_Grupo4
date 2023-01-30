def frameadjustment(self, distance_threshold=0.1, ransac_n=4, num_iterations=120):
        
        frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=1, origin=np.array([0, 0, 0]))

        # -------- Segmentation ----------
        
        # Segmentation Vars
        table_pcd = self.pcd
        num_planes = 2
        detected_plane_idx = []
        detected_plane_d = []
        

        while True:
            # Plane Segmentation
            plane_model, inliers = table_pcd.segment_plane(distance_threshold, ransac_n, num_iterations)

            # Plane Model
            [a, b, c, d] = plane_model
            
            # If there is a plane that have de negative y, will be necessary make one more measurement/segmentation 
            if b < 0:
                num_planes = 3

            # Inlier Cloud
            inlier_cloud = table_pcd.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            
            # Segmetation pcd update
            outlier_cloud = table_pcd.select_by_index(inliers, invert=True)
            table_pcd = outlier_cloud

            # Append detected plane
            if b > 0:
                detected_plane_idx.append(inlier_cloud)
                detected_plane_d.append(d)

            # Condition to stop pcd segmetation (2 measurments/segmentations)
            if len(detected_plane_idx) >= num_planes: 
                num_planes = 2
                break
        
        # Find idx of the table plane 
        d_max_idx = min(range(len(detected_plane_d)), key=lambda i: abs(detected_plane_d[i]-0))
        table_pcd = detected_plane_idx[d_max_idx]
        
        # -------- Table Plane Clustering ----------
        # Clustering 
        cluster_idx = np.array(table_pcd.cluster_dbscan(eps=0.08, min_points=50))
        objects_idx = list(set(cluster_idx))


        if cluster_idx.any() == -1:
            objects_idx.remove(-1)  
        
        # -------- Planes Caracterization ----------

        # Colormap
        colormap = cm.Set2(list(range(0,len(objects_idx))))

        # Caracterize all planes found to proceed to table detection/isolation 
        objects=[]
        for object_idx in objects_idx:
            
            object_point_idx = list(locate(cluster_idx, lambda X: X== object_idx))
            object_points = table_pcd.select_by_index(object_point_idx)
            object_center = object_points.get_center()

            # Create a dictionary to represent all planes
            d = {}
            d['idx'] = str(objects_idx)
            d['points'] = object_points
            d['color'] = colormap[object_idx, 0:3]
            d['points'].paint_uniform_color(d['color'])
            d['center'] = object_center
            
            objects.append(d)

        # -------- Table Selection ----------

        # The table is deteted with the comparison between the coordinates of centers (pcd and frame) and need to have more than 10000 points
        tables_to_draw=[]
        minimum_mean_xy = 1000
        
        for object in objects:
            tables_to_draw.append(object['points'])
            mean_x = object['center'][0]
            mean_y = object['center'][1]
            mean_z = object['center'][2]
            mean_xy = abs(mean_x) + abs(mean_y)
            if mean_xy < minimum_mean_xy:
                minimum_mean_xy = mean_xy
                if len(np.asarray(object['points'].points)) > 12000:
                    
                    self.table_cloud = object['points']
                    
        
        center = self.table_cloud.get_center() # visual center of the table
        tx, ty, tz = center[0], center[1], center[2] 

        return(-tx, -ty, -tz)