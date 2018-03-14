#!/usr/bin/env python3
'''Records scans to a given file in the form of numpy array.
Usage example:

$ ./record_scans.py out.npy'''
import sys
import numpy as np
from rplidar.rplidar import RPLidar
import pdb
import time
from numpy.random import choice
import random

PORT_NAME = '/dev/ttyUSB0'

class Obstacle():
    """
    Dynamic or static rectangular obstacle. It is assumed that dynamic objects are under constant acceleration.
    E.g. moving vehicle, parked vehicle, wall
    """
    def __init__(self, centroid, dx, dy, angle=0, vel=[1, 0], acc=[0, 0]):
        """
        :param centroid: centroid of the obstacle
        :param dx: length of the vehicle >=0
        :param dy: width of the vegicle >= 0
        :param angle: anti-clockwise rotation from the x-axis
        :param vel: [x-velocity, y-velocity], put [0,0] for static objects
        :param acc: [x-acceleration, y-acceleration], put [0,0] for static objects/constant velocity
        """
        self.centroid = centroid
        self.dx = dx
        self.dy = dy
        self.angle = angle
        self.vel = vel #moving up/right is positive
        self.acc = acc
        self.time = 0 #time is incremented for every self.update() call

    def __get_points(self, centroid):
        """
        :return A line: ((x1,y1,x1',y1'))
                or four line segments: ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (x4,y4,x4',y4'))
        """
        dx_cos = self.dx*np.cos(self.angle)
        dx_sin = self.dx*np.sin(self.angle)
        dy_sin = self.dy*np.sin(self.angle)
        dy_cos = self.dy*np.cos(self.angle)

        # BR_x = centroid[0] + 0.5*(dx_cos + dy_sin) #BR=Bottom-right
        # BR_y = centroid[1] + 0.5*(dx_sin - dy_cos)
        # BL_x = centroid[0] - 0.5*(dx_cos - dy_sin)
        # BL_y = centroid[1] - 0.5*(dx_sin + dy_cos)
        # TL_x = centroid[0] - 0.5*(dx_cos + dy_sin)
        # TL_y = centroid[1] - 0.5*(dx_sin - dy_cos)
        # TR_x = centroid[0] + 0.5*(dx_cos - dy_sin)
        # TR_y = centroid[1] + 0.5*(dx_sin + dy_cos)

        BR_x = centroid[0] + 0.5*(dx_cos + dy_sin) #BR=Bottom-right
        BR_y = centroid[1] + 0.5*(dx_sin - dy_cos)
        BL_x = centroid[0] - 0.5*(dx_cos - dy_sin)
        BL_y = centroid[1] - 0.5*(dx_sin + dy_cos)
        TL_x = centroid[0] - 0.5*(dx_cos + dy_sin)
        TL_y = centroid[1] - 0.5*(dx_sin - dy_cos)
        TR_x = centroid[0] + 0.5*(dx_cos - dy_sin)
        TR_y = centroid[1] + 0.5*(dx_sin + dy_cos)        
        seg_bottom = (BL_x, BL_y, BR_x, BR_y)
        seg_left = (BL_x, BL_y, TL_x, TL_y)

        if self.dy == 0: #if no height
            return (seg_bottom,)
        elif self.dx == 0: # if no width
            return (seg_left,)
        else: #if rectangle
            seg_top = (TL_x, TL_y, TR_x, TR_y)
            seg_right = (BR_x, BR_y, TR_x, TR_y)
            return (seg_bottom, seg_top, seg_left, seg_right)

    def __get_points_old(self, centroid):
        """
        :return A line: ((x1,y1,x1',y1'))
                or four line segments: ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (x4,y4,x4',y4'))
        """
        seg_bottom = (centroid[0] - self.dx/2, centroid[1] - self.dy/2, centroid[0] + self.dx/2, centroid[1] - self.dy/2)
        seg_left = (centroid[0] - self.dx/2, centroid[1] - self.dy/2, centroid[0] - self.dx/2, centroid[1] + self.dy/2)

        if self.dy == 0: #if no height
            return (seg_bottom,)
        elif self.dx == 0: # if no width
            return (seg_left,)
        else: #if rectangle
            seg_top = (centroid[0] - self.dx/2, centroid[1] + self.dy/2, centroid[0] + self.dx/2, centroid[1] + self.dy/2)
            seg_right = (centroid[0] + self.dx/2, centroid[1] - self.dy/2, centroid[0] + self.dx/2, centroid[1] + self.dy/2)
            return (seg_bottom, seg_top, seg_left, seg_right)

    def update(self):
        """
        :return: updated centroid
        """
        disp_x = self.vel[0]*self.time + 0.5*self.acc[0]*(self.time**2) #s_x = ut + 0.5at^2
        disp_y = self.vel[1]*self.time + 0.5*self.acc[1]*(self.time**2) #s_y = ut + 0.5at^2
        self.time += 1 #time is incremented for every self.update() call
        return self.__get_points(centroid=[self.centroid[0] + disp_x, self.centroid[1] + disp_y])

# def run(path):
#     '''Main function'''
#     lidar = RPLidar(PORT_NAME)
#     data = []
#     try:
#         print('Recording measurments... Press Crl+C to stop.')
#         for scan in lidar.iter_scans():
#             data=np.array(scan)
#             pritn(data[2])
#     except KeyboardInterrupt:
#         print('Stoping.')
#     lidar.stop()
#     lidar.disconnect()
#     np.save(path, np.array(data))

# if __name__ == '__main__':
#     run(sys.argv[1])
def connect_segments(segments, resolution = 10):
    """
    :param segments: start and end points of all segments as ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (...))
           step_size : resolution for plotting
    :return: stack of all connected line segments as (X, Y)
    """

    for i, seg_i in enumerate(segments):
        if seg_i[1] == seg_i[3]: #horizontal segment
            x = np.arange(min(seg_i[0],seg_i[2]), max(seg_i[0],seg_i[2]), resolution)
            y = seg_i[1]*np.ones(len(x))
        elif seg_i[0] == seg_i[2]: #vertical segment
            y = np.arange(min(seg_i[1],seg_i[3]), max(seg_i[1],seg_i[3]), resolution)
            x = seg_i[0]*np.ones(len(y))
        else: # gradient exists
            m = (seg_i[3] - seg_i[1])/(seg_i[2] - seg_i[0])
            c = seg_i[1] - m*seg_i[0]
            x = np.arange(min(seg_i[0],seg_i[2]), max(seg_i[0],seg_i[2]), resolution)
            y = m*x + c

        obs = np.vstack((x, y)).T
        if i == 0:
            connected_segments = obs
        else:
            connected_segments = np.vstack((connected_segments, obs))

    return connected_segments


def get_intersection(a1, a2, b1, b2) :
    """
    :param a1: (x1,y1) line segment 1 - starting position
    :param a2: (x1',y1') line segment 1 - ending position
    :param b1: (x2,y2) line segment 2 - starting position
    :param b2: (x2',y2') line segment 2 - ending position
    :return: point of intersection, if intersect; None, if do not intersect
    #adopted from https://github.com/LinguList/TreBor/blob/master/polygon.py
    """

    def perp(a) :
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    intersct = np.array((num/denom.astype(float))*db + b1) #TODO: check divide by zero!

    delta = 1e-3
    condx_a = min(a1[0], a2[0])-delta <= intersct[0] and max(a1[0], a2[0])+delta >= intersct[0] #within line segment a1_x-a2_x
    condx_b = min(b1[0], b2[0])-delta <= intersct[0] and max(b1[0], b2[0])+delta >= intersct[0] #within line segment b1_x-b2_x
    condy_a = min(a1[1], a2[1])-delta <= intersct[1] and max(a1[1], a2[1])+delta >= intersct[1] #within line segment a1_y-b1_y
    condy_b = min(b1[1], b2[1])-delta <= intersct[1] and max(b1[1], b2[1])+delta >= intersct[1] #within line segment a2_y-b2_y
    if not (condx_a and condy_a and condx_b and condy_b):
        intersct = None #line segments do not intercept i.e. interception is away from from the line segments

    return intersct

def get_laser_ref(segments, angle, xy_robot,max_dist=10000):
    """
    :param segments: start and end points of all segments as ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (...))
           realm_in_radians: sight of the robot - typically pi or 4/3*pi
           n_reflections: resolution=realm_in_radians/n_reflections
           max_dist: max distance the robot can see. If no obstacle, laser end point = max_dist
           xy_robot: robot's position in the global coordinate system
    :return: 1xn_reflections array indicating the laser end point
    """

    dist_theta = max_dist
    theta=angle
    x_pos = max_dist*np.cos(theta)+xy_robot[0]
    y_pos = max_dist*np.sin(theta)+xy_robot[1]
    xy_ij_max = np.array([x_pos, y_pos]) # max possible distance
    for seg_i in segments:
        xy_i_start, xy_i_end = np.array(seg_i[:2]), np.array(seg_i[2:]) #starting and ending points of each segment
        #pdb.set_trace()
        intersection=get_intersection(xy_i_start, xy_i_end, xy_robot, xy_ij_max)
        #TODO: when the robot is moving
        if intersection is not None: #if the line segments intersect
            r = np.sqrt( (intersection[0]-xy_robot[0])**2 + (intersection[1]-xy_robot[1])**2 )
            if r < dist_theta:
                dist_theta = r
    return dist_theta

obs=[[[0,0],[0,0]] for i in range(20)]
obs[0]=[[2200,0],[2500,800]]
obs[1]=[[3700,1200],[4000,2000]]
obs[2]=[[1500,1800],[2700,2100]]
obs[3]=[[0,3100],[2000,3400]]
for i in range(4):
    t=obs[i]
    obs[i+4]=[[5000-t[1][0],8000-t[1][1]],
                                [5000-t[0][0],8000-t[0][1]]]
obs[8]=[[-300,-300],[5300,0]]
obs[9]=[[-300,-300],[0,8300]]
obs[10]=[[5000,-300],[5300,8300]]
obs[11]=[[-300,8000],[5300,8300]]
    
obs=np.array(obs)
all_obstacles=[]
for i in range(12):
    all_obstacles.append(Obstacle(centroid=[(obs[i][1][0]+obs[i][0][0])/2, 
            (obs[i][1][1]+obs[i][0][1])/2], dx=obs[i][1][0]-obs[i][0][0], 
            dy=obs[i][1][1]-obs[i][0][1], angle=0, vel=[0, 0], acc=[0, 0]))

#lidar = RPLidar(PORT_NAME)
robot_pos=np.array([100, 100])
particle_num=100
dis_pos=[0,0]
particle_order=[i for i in range(particle_num)]
particle_weight=[0 for i in range(particle_num)]
particle_pos=[]
for i in range(particle_num):
    particle_pos.append([random.uniform(0,5000),random.uniform(0,8000)])
    #print(particle_pos)

# for scan in lidar.iter_scans():
#     data=np.array(scan)
all_obstacle_segments = []
for obs_i in all_obstacles:
    all_obstacle_segments += obs_i.update()
data=[]
for i in range(100):
    t=random.uniform(0,np.pi*2)
    data.append([0,t,get_laser_ref(all_obstacle_segments,t,robot_pos)])

for _  in range(10000):
    sum_error=0
    for num in range(particle_num):
        particle_pos[num][0]+=dis_pos[0]
        particle_pos[num][1]+=dis_pos[1]

        error=0
        for i in range(np.shape(data)[0]):
            angle=data[i][1]/360*np.pi*2
            error+=(get_laser_ref(all_obstacle_segments,angle,particle_pos[num])-data[i][2])**2
        error=error/np.shape(data)[0]
        particle_weight[num]=error
        sum_error+=error
    particle_weight=particle_weight/sum_error

    new_particle=[]
    for num in range(particle_num):
        t=choice(particle_order, p=particle_weight)
        new_particle.append(particle_pos[t])
    particle_pos=new_particle

    x=0
    y=0
    for i in range(particle_num):
        x=x+particle_pos[i][0]
        y=y+particle_pos[i][1]
    x=x/particle_num
    y=y/particle_num
    print(x,y)
    #print(np.shape(data))
    #time.sleep(1)
        
