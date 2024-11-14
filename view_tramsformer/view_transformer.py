import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        # actual meters 
        court_width = 68
        court_length = 23.32 

        # found values by guessing until looked right
        # trapezoid
        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 950]
        ])
        # rectangle
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_width, court_length],
            [court_width, 0]
        ])

        # convert to smth numpy understands
        self.pixel_vertices = self.pixel_vertices.astype(np.float)
        self.target_vertices = self.target_vertices.astype(np.float)

        # transform perspective , wala
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    # calculate position relative to actual court in meters
    def transform_point(self, point):
        x = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, x, False) >= 0
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)

        return transform_point.reshape(-1, 2)


    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    psoition = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed


    




