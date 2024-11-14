def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0] # x2-x1

def measure_dist(x, y):
    return ((x[0] - y[0]) **2 + (x[1]-y[1])**2)**0.5

def measure_xy_distance(x, y):
    return x[0] - y[0], x[1] - y[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2), int(y2)

