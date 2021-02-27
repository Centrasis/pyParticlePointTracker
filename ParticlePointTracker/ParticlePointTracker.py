
from joblib import Parallel, delayed
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import angle
import numpy as np
import cv2
import math
from typing import Callable, Dict, Iterable, Iterator, Tuple, List
#from pfilter import ParticleFilter, gaussian_noise, squared_error, independent_sample
from scipy.stats import norm, gamma, uniform
from matplotlib import pyplot as plt
from .Features import ParticleFeature


class ParticlePointTracker(object):
    """
        Simple Particle Tracker that takes serveral features.
        The result after each step will be stored in self.resulting_points
    """
    particles_error: np.ndarray = None
    particle_count = 100
    regionSize: int = 3
    debug_img: np.ndarray = None
    last_points = []
    features: List[ParticleFeature] = []
    movement_directions: np.ndarray = None
    filter_repeat = 0
    max_filter_repeats = 6
    thread_count = 8
    search_radius = 0.2
    flow_map: np.ndarray = None
    features_generator: Callable[[], List[ParticleFeature]]
    point_trackers = []
    resulting_points: np.ndarray = None

    def __init__(self, features_generator: Callable[[], List[ParticleFeature]], regionSize = 3, particles_count = 200, one_tracker_per_particle = True) -> None:
        super().__init__()
        self.features_generator = features_generator
        self.particle_count = particles_count
        self.regionSize = regionSize
        self.features = []
        self.last_particles = None
        self.last_points = []
        self.filter_repeat = 0
        self.max_filter_repeats = ParticlePointTracker.max_filter_repeats
        self.thread_count = ParticlePointTracker.thread_count
        self.one_tracker_per_particle = one_tracker_per_particle
        self.__reset__()    

    def __reset__(self):
        self.current_img = None
        self.particles_weight = None
        self.particles = None
        self.features = []
        self.point_trackers = []
        self.search_radius = ParticlePointTracker.search_radius
        self.flow_map = None
        self.features = self.features_generator()
        self.resulting_points = None

    def __particle_observation(self, particles: np.ndarray):
        """
            parent function to observe all enabled features
        """
        #[anzahl partikel][anzahl der Punkte pro Partikel][x,y position]
        def observe_per_particle(particle: np.ndarray):
            for j, pos in enumerate(particle):
                for b, f in enumerate(self.features):
                    f.observe(self.current_img[int(pos[1] - self.regionSize // 2):int(pos[1] + self.regionSize // 2),
                                                            int(pos[0] - self.regionSize // 2):int(pos[0] + self.regionSize // 2)], 
                                            particle, j)
        # calc observationmatrix for all enabled features in different threads
        Parallel(n_jobs=self.thread_count, prefer="threads")(
            delayed(observe_per_particle)(particle) for i, particle in enumerate(particles)
        )

    def __particles_error(self, particles):
        """
            parent function for calculate the error of each feature
            :param particles: particles per image with pos[x][y]
        """
        #calculate the error for each particle 
        def error_per_particle(error: np.ndarray, particle_idx: int, particle: np.ndarray):
            errors = np.zeros((particles.shape[1]), dtype=np.float)
            for j in range(particles.shape[1]):
                for fi, f in enumerate(self.features):
                    errors[j] = errors[j] + f.error(self.current_img, particle, j)

            normalize = 0.0
            for _, f in enumerate(self.features):
                normalize += f.importance

            std_dev = np.std(errors) * 2.0
            #error matrix per particle
            error[particle_idx] = (np.sum(errors) + std_dev) / (particle.shape[1] * (normalize + 1))                     

        error = np.zeros((particles.shape[0],), dtype=np.float)
        #calculate the error of each feature in different threads
        Parallel(n_jobs=self.thread_count, prefer="threads")(
            delayed(error_per_particle)(error, particle_idx, particle) for particle_idx, particle in enumerate(particles)
        )

        error = error.transpose()
        #return the error matrix of each particle
        return error

    def resample(self, particles: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
            Search for new particles based on the optical flow
            :param particles: current particles
            :param scale: error as scale factor to influence the search area if needed
        """
        k_resamples = 10
        resample_base = np.argpartition(scale, k_resamples)[0:k_resamples]
        part_size = particles.shape[0] // k_resamples
        old_particles = particles[resample_base].copy()
        search = (
            self.search_radius / 2 * self.current_img.shape[1],
            self.search_radius / 2 * self.current_img.shape[0]
        )
        for r in range(k_resamples):
            for i in range(part_size * r, (part_size * (r + 1)) if r < k_resamples - 1 else particles.shape[0]):  # iterate over each particle
                # scale the error using a logistic function to dampen the search area
                s = 2.0 / (1 + math.exp(-scale[resample_base[r]] * 5 + 5) + 0.2)
                for j in range(particles.shape[1]):   # iterate over each point that describes the shape
                    sample_orig = (
                        old_particles[r][j][0],
                        old_particles[r][j][1]
                    )
                    if not (self.flow_map is None):
                        # calc optical flow to constrain the particle generation according to the frames change
                        box = (
                            int(sample_orig[1] - self.regionSize // 2),
                            int(sample_orig[1] + self.regionSize // 2), 
                            int(sample_orig[0] - self.regionSize // 2),
                            int(sample_orig[0] + self.regionSize // 2)
                        )
                        dominant_dir = self.flow_map[box[0]:box[1], box[2]:box[3]]
                        xs = np.abs(dominant_dir[...,0])
                        ys = np.abs(dominant_dir[...,1])
                        ydir = abs(np.sum(ys) / ys.shape[0])
                        xdir = abs(np.sum(xs) / xs.shape[0])
                        search = (
                            xdir,
                            ydir
                        )
                    # draw the particles coordinates from normal distibution around the current point and in dominant direction
                    particles[i][j][0] = np.clip(np.random.normal(loc=sample_orig[0], scale=s * search[0]), 0 + self.regionSize // 2, self.current_img.shape[1] - self.regionSize // 2)
                    particles[i][j][1] = np.clip(np.random.normal(loc=sample_orig[1], scale=s * search[1]), 0 + self.regionSize // 2, self.current_img.shape[0] - self.regionSize // 2)
        #return the new particles
        return particles

    def update(self):
        """
            Update the particles based on error
        """
        #get new particles based on error
        if not (self.particles_weight is None):
            self.particles = self.resample(self.particles, self.particles_error)
        # observe the particles
        self.__particle_observation(self.particles)
        # calc the error of particles 
        self.particles_error = self.__particles_error(self.particles)
        #for i in range(self.particles.shape[0]):
        #    for j in range(self.particles.shape[1]):
        #        self.debug_img = cv2.circle(self.debug_img, (int(self.particles[i][j][0]),int(self.particles[i][j][1])), int(self.regionSize / 2),(0, int(255 * (1-self.particles_error[i])),0),2)
        err_sum = np.sum(self.particles_error)

        if err_sum != 0.0:
            self.particles_weight = np.divide(self.particles_error,err_sum)
        else:
            self.particles_weight = np.ones((self.particles_error.shape[0]), dtype=np.float) / self.particles_error.shape[0]

    def evaluate(self):
        """
            Evaluates the best and mean particle and their errors after an update.
        """
        arg_min = np.argmin(self.particles_error)
        self.best_particle = self.particles[arg_min]
        self.best_error = self.particles_error[arg_min]
        self.mean_particle = np.zeros((self.particles.shape[1], 2), dtype=np.float)
        for pi in range(self.particles.shape[0]):
            for p in range(self.particles.shape[1]):
                self.mean_particle[p] = self.mean_particle[p] + self.particles[pi][p] * self.particles_weight[pi]
        self.mean_error = np.sum(self.particles_error) / self.particle_count
         
    def set_initial_points(self, points: np.ndarray):
        """
            :param points: shape(n, 2) with content -> [[x1,y1], [x2,y2], ...]
        """
        self.initial_points = points.copy()

    def step(self, image: np.ndarray):
        """
            Start the PaticlePointTracker
        """
        old_points = self.initial_points if self.resulting_points is None else self.resulting_points

        self.last_points = old_points
        # run  one particle filter per point, iterate over each point
        if self.one_tracker_per_particle and self.last_points.shape[0] > 1:
            if len(self.point_trackers) == 0:
                for i in range(self.last_points.shape[0]):
                    self.point_trackers.append(ParticlePointTracker(self.features_generator, self.regionSize, self.particle_count, False))
                for i in range(self.last_points.shape[0]):
                    source = self.last_points[i:i+1]
                    self.point_trackers[i].set_initial_points(source)
                    self.point_trackers[i].step(image)
                return

            self.resulting_points = old_points
            for i in range(self.last_points.shape[0]):
                self.point_trackers[i].step(image)
                self.resulting_points[i] = self.point_trackers[i].resulting_points[0]
            return

        if not (self.current_img is None):
            # calc the opical flow over the entire image -> get a metric on how broad the filter should search per pixel
            self.flow_map = cv2.calcOpticalFlowFarneback(cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)

        self.current_img = image

        if self.particles is None:
            self.particle_count *= old_points.shape[0]
            self.particles = np.zeros((self.particle_count, old_points.shape[0], 2), dtype=np.float)
            for i in range(self.particles.shape[0]):  # iterate over each particle
                for j in range(self.particles.shape[1]):
                    x,y = np.ravel(old_points[j])
                    self.particles[i][j][0] = x
                    self.particles[i][j][1] = y
            
            self.__particle_observation(self.particles[0:1])

            self.orig_points = self.last_points.copy()
            self.resulting_points = self.last_points.copy()      
            return
        
        self.filter_repeat = 0
        end_ms = np.zeros((self.last_points.shape[0], 2))
        last_mean_err = 0.0
        # ground truth set, then go for new particles, run in loop for max_filter_repeats
        while True:
            #self.debug_img = image.copy()
            #for i in range(old_points.shape[0]):
            #    self.debug_img = cv2.circle(self.debug_img, (int(old_points[i][0][0]),int(old_points[i][0][1])), int(self.regionSize / 2),(0, 0, 255, 2))
            #    self.debug_img = cv2.putText(self.debug_img, str(i), (int(old_points[i][0][0]) - self.regionSize // 4,int(old_points[i][0][1]) + self.regionSize // 4), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0))
            # update the particles in observation and error
            self.update()
            # evaluate the results of particles error for getting the mean error/best error
            self.evaluate()
            #ms = self.mean_particle #self.filter.mean_state
            #for j in range(ms.shape[0]):
            #    self.debug_img = cv2.circle(self.debug_img, (int(ms[j][0]),int(ms[j][1])), int(self.regionSize / 2),(255,0,0),2)
            bs = self.best_particle
            #for j in range(bs.shape[0]):
            #    self.debug_img = cv2.circle(self.debug_img, (int(bs[j][0]),int(bs[j][1])), int(self.regionSize / 2),(0,255,0),2)
            #cv2.imshow("Debug Particle", self.debug_img)
            #cv2.waitKey(5)

            end_ms += bs * (1-self.best_error)
            last_mean_err = last_mean_err + (1-self.best_error)

            self.filter_repeat = self.filter_repeat + 1
            if self.filter_repeat > self.max_filter_repeats:
                break
        if last_mean_err > 0:
            end_ms /= last_mean_err
        else:
            end_ms /= self.filter_repeat

        self.resulting_points = end_ms

        if self.last_particles is None:
            self.last_particles = self.particles