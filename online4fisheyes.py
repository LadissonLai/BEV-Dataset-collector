#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import time
import cv2
from surroundBEV import BevGenerator, padding, BEV_HEIGHT, BEV_WIDTH

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random

try:
    import pygame
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()  # 返回下一帧的ID，执行world.on_tick的回调函数。
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

def trans_cv2_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # BGR
    return array

def draw_image(surface, image, pos=(0, 0), blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # BGR
    x, y = array.shape[0:2]
    array = cv2.resize(array, (int(y / 4), int(x / 4)))
    array = array[:, :, ::-1]  # RGB
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, pos)


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def main():
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (1000, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    
    # 设置同步模式
    # fps=20
    # settings = world.get_settings()
    # settings.synchronous_mode = True
    # settings.fixed_delta_seconds = 1/fps
    # world.apply_settings(settings)

    try:
        m = world.get_map()
        
        # loc = carla.Location(x=-17.5, y=26.5, z=-2.0)
        loc = carla.Location(x=-15.5, y=27.9, z=-2.0) # 这个位置需要保留，测试bev泊车的起点
        rot = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        start_pose = carla.Transform(loc, rot)
        
        # start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.tesla.model3')),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        # fisheyes transforms
        Attachment = carla.AttachmentType
        camera_transforms = [
            (carla.Transform(carla.Location(x=2.30, z=1.0),
             carla.Rotation(pitch=-20.0)), Attachment.Rigid),  # 前
            (carla.Transform(carla.Location(x=0.0, y=-1.05, z=1.0),
             carla.Rotation(yaw=-90, pitch=-20.0)), Attachment.Rigid),  # 左
            (carla.Transform(carla.Location(x=0.0, y=1.05, z=1.0),
                             carla.Rotation(yaw=90, pitch=-20.0)), Attachment.Rigid),  # 右边
            (carla.Transform(carla.Location(x=-2.30, z=1.0),
                             carla.Rotation(yaw=180, pitch=-20.0)), Attachment.Rigid),  # 后面
        ]
        bp_fisheye = blueprint_library.find('sensor.camera.huawei_fisheye')
        bp_fisheye.set_attribute('image_size_x', str(1920))
        bp_fisheye.set_attribute('image_size_y', str(1080))

        # spawn 4 fisheyes
        front_fisheye = world.spawn_actor(
            bp_fisheye, camera_transforms[0][0], attach_to=vehicle, attachment_type=camera_transforms[0][1])
        left_fisheye = world.spawn_actor(
            bp_fisheye, camera_transforms[1][0], attach_to=vehicle, attachment_type=camera_transforms[1][1])
        right_fisheye = world.spawn_actor(
            bp_fisheye, camera_transforms[2][0], attach_to=vehicle, attachment_type=camera_transforms[2][1])
        back_fisheye = world.spawn_actor(
            bp_fisheye, camera_transforms[3][0], attach_to=vehicle, attachment_type=camera_transforms[3][1])
        actor_list.append(front_fisheye)
        actor_list.append(left_fisheye)
        actor_list.append(right_fisheye)
        actor_list.append(back_fisheye)
        
        world.tick()

        # car = cv2.imread('./data/car.jpg')
        # car = padding(car, BEV_WIDTH, BEV_HEIGHT)
        # Create a synchronous mode context.
        with CarlaSyncMode(world, front_fisheye, left_fisheye, right_fisheye, back_fisheye, fps=30) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                # snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)
                snapshot, front, left, right, back = sync_mode.tick(
                    timeout=2.0)

                # Choose the next waypoint and update the car location.
                waypoint = random.choice(waypoint.next(1.5))
                vehicle.set_transform(waypoint.transform)

                # image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                draw_image(display, front, (0, 0))
                draw_image(display, back, (490, 0))
                draw_image(display, left, (0, 280))
                draw_image(display, right, (490, 280))
                # draw_image(display, image_semseg, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' %
                                clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' %
                                fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()

                
                front = trans_cv2_image(front)
                back = trans_cv2_image(back)
                left = trans_cv2_image(left)
                right = trans_cv2_image(right)
                bev = BevGenerator()
                # surround = bev(front, back, left, right, car)
                surround = bev(front, back, left, right, None)
                cv2.imwrite('./co_images/bev/'+str(snapshot.timestamp.frame)+'_bev.png', surround)

                cv2.namedWindow(
                    'surround', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.imshow('surround', surround)
                k = cv2.waitKey(1)
                if k == ord('q'):  # 按q键退出
                    cv2.destroyAllWindows()
                    break
                

    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
