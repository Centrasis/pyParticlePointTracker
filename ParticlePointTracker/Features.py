import numpy as np
import cv2
import math
from enum import IntEnum
from skimage.feature import greycomatrix, greycoprops

class HIST_HSV_FEATURE(IntEnum):
    EUCLIDEAN_DIST = 0
    CORRELATION = 1
    CHISQR = 2
    INTERSECTION = 3
    BHATTACHARYYA = 4

class FeatureHelper:
    """
        Helper class for Particlefilter features
    """
    @staticmethod
    def getDistanceMatrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """
            calc the distancematrix of two points
        """
        mat = np.zeros((points1.shape[0], points2.shape[0]), dtype=np.float)

        for i, p1 in enumerate(points1):
            if p1.shape[0] == 1:
                p1 = p1[0]
            for j, p2 in enumerate(points2):
                if p2.shape[0] == 1:
                    p2 = p2[0]
                diff = p1 - p2
                mat[i, j] = np.linalg.norm(diff)

        return mat


class ParticleFeature(object):
    """
        Parent-class for ParticleFilter features
    """
    max_observation_channels: int = 256
    importance = 1.0

    def __init__(self, ground_truth, importance) -> None:
        super().__init__()
        self.ground_truth = ground_truth
        self.importance = importance

    def observe(self, image: np.ndarray, particle: np.ndarray, point_index:int, **kwargs) -> None:
        pass

    def error(self, image: np.ndarray, particle: np.ndarray, point_index:int, **kwargs) -> float:
        pass

    def max_error() -> float:
        return 1.0

class GLCMFeature(ParticleFeature):
    """
        Gray-Level-Co-Occurence-Matrix feature for Particlefilter
    """
    measures = ["correlation", "homogeneity", "energy"] # "dissimilarity"

    def __init__(self, ground_truth, **kwargs) -> None:
        super().__init__(ground_truth, kwargs.get("importance", 1.0))
        self.ground_truths = {}
        self.last_observ = {}
        

    def observe(self, image: np.ndarray, particle: np.ndarray, point_index:int, **kwargs) -> None:
        """
            calc the Gray-Level-Co-Occurence-Matrix for each particle
            :param particle: particles per image with pos[x][y]
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = greycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        if not (point_index in self.ground_truths.keys()):
            self.ground_truths[point_index] = glcm

        self.last_observ[point_index] = glcm

    def getNormCoeff(self, measure: str, mat: np.ndarray) -> float:
        normCoeff = 1.0
        if measure == "dissimilarity":
            normCoeff = 0.0
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    normCoeff += abs(i - j)
        return normCoeff

    def error(self, image: np.ndarray, particle: np.ndarray, point_index:int, **kwargs) -> float:
        """
            weigth the matches of SIFT features -> through BF-Matcher take only the identical points
            :param particle: particles per image with pos[x][y]
        """
        ground = self.ground_truths[point_index]
        particle = self.last_observ[point_index]
        error = 0.0

        for measure in GLCMFeature.measures:
            normCoeff = self.getNormCoeff(measure, ground)
            ground_meas = greycoprops(ground, measure).ravel()[0] / normCoeff
            particle_meas = greycoprops(particle, measure).ravel()[0] / normCoeff
            error += abs(ground_meas - particle_meas)

        return error / len(GLCMFeature.measures) * self.importance


class SIFTFeature(ParticleFeature):
    """
        SIFT Feature Class for Particle Filter
    """
    def __init__(self, ground_truth, **kwargs) -> None:
        super().__init__(ground_truth, kwargs.get("importance", 1.0))
        self.detector = cv2.SIFT_create()
        # BFMatcher with default params
        self.ground_truths = {}
        self.last_observ = {}
        self.bf = cv2.BFMatcher()
        self.matching_channels = 0
        self.metrics = kwargs.get("metric", HIST_HSV_FEATURE.EUCLIDEAN_DIST)

    def observe(self, image: np.ndarray, particle: np.ndarray, point_index:int, **kwargs) -> None:
        """
            calc the SIFT Features of Gray Image for each particle
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self.detector.detectAndCompute(image,None)
        if not (point_index in self.ground_truths.keys()):
            self.ground_truths[point_index] = (des1, kp1, image)

        self.diagonal = np.linalg.norm(image.shape[0:2]) / 2

        self.last_observ[point_index] = (des1, kp1, image)

    def error(self, image: np.ndarray, particle: np.ndarray, point_index:int, **kwargs) -> float:
        """
            weigth the matches of SIFT features -> through BF-Matcher take only the identical points
        """
        des_ground = self.ground_truths[point_index][0]
        des_particle = self.last_observ[point_index][0]
        kp_ground = self.ground_truths[point_index][1]
        kp_particle = self.last_observ[point_index][1]
        dist = 0.0 if not (des_particle is None) else 1.0

        if dist == 0.0:
            try:
                # match the points between current particles and groundtruth
                matches = self.bf.match(des_particle, des_ground)
                for m in matches:
                    dist = dist + math.sqrt((kp_ground[m.trainIdx].pt[0] - kp_particle[m.queryIdx].pt[0]) ** 2 + (kp_ground[m.trainIdx].pt[1] - kp_particle[m.queryIdx].pt[1]) ** 2)

                if len(matches) > 0:
                    dist += (self.diagonal * (des_ground.shape[0] - len(matches)))
                    dist /= (self.diagonal * len(matches))
                    dist = np.clip(dist, 0, 1)
                else:
                    dist = 1.0
            except Exception as e:
                dist = 1.0

        #if point_index == 0:
        #    match_img = cv2.drawMatches(
        #                self.ground_truths[point_index][2], self.ground_truths[point_index][1],
        #                self.last_observ[point_index][2], self.last_observ[point_index][1],
        #                matches[:min(10, len(matches))], self.last_observ[point_index][2].copy(), flags=0)
        #    cv2.imshow("Matches: " + str(point_index), match_img)

        # weight the SIFT Feature with importance factor for global comparison with other features    
        return dist * self.importance


class ORBFeature(SIFTFeature):
    """
        ORB Feature for Particle Filter
    """
    def __init__(self, ground_truth, **kwargs) -> None:
        super().__init__(ground_truth, **kwargs)
        #create ORB Detector with default params
        self.detector = cv2.ORB_create()


class TemplateFeature(ParticleFeature):
    """
        Template Feature for Particle Filter
    """
    def __init__(self, ground_truth, **kwargs) -> None:
        super().__init__(ground_truth, kwargs.get("importance", 1.0))
        self.template = None

    def observe(self, image: np.ndarray, particle: np.ndarray, point_index:int, **kwargs) -> None:
        """
            calc the Template Features of Gray Image for each particle
            :param image: part of image around particle
            :param particle: particle on image with pos[x][y]
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.template is None:
            self.template = image

        self.current_img = image
        

    def error(self, image: np.ndarray, particle: np.ndarray, point_index:int, **kwargs) -> float:
        """
            weigth the matches of Template features -> through BF-Matcher take only the identical points
            :param image: hole image
            :param obervation: wanted template
            :param particle: particle on image with pos[x][y]
            :param point_index: idx of point
            :param ground_truth: ground truth
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # find the template on image
        res = cv2.matchTemplate(image, self.template, cv2.TM_SQDIFF_NORMED)

        def variance_of_laplacian(image):
            # compute the Laplacian of the image and then return the focus
            # measure, which is simply the variance of the Laplacian
            return cv2.Laplacian(image, cv2.CV_64F).var()
        # value of blurines in the template to verify the confidence
        blur = variance_of_laplacian(self.current_img)

        # check for blurines and if template inside the image, else clip it
        return res[np.clip(int(particle[0][1]), 0, image.shape[0] - self.current_img.shape[0]), np.clip(int(particle[0][0]), 0, image.shape[1] - self.current_img.shape[1])] * self.importance * (1.0 if blur > 200 else 0.5)

class HistogrammHSVFeature(ParticleFeature):
    """
        Histogram Feature based on HSV Colorspace for Particle Filter
    """
    def __init__(self, ground_truth, **kwargs) -> None:
        """
            init the HistogramHSVFeature
            :param ground_truth: histogram on first original image
            :param **kwargs: 'importance' to weight the feature impact; 'metric' choose the metric of histogram difference; 'channel_idx' select the channel of histogram
        """
        super().__init__(ground_truth, kwargs.get("importance", 1.0))
        self.channel_idx = kwargs.get("channel_idx")
        self.metrics = kwargs.get("metric", HIST_HSV_FEATURE.EUCLIDEAN_DIST)
        self.bins = kwargs.get("bins")
        self.observation = dict()
        self.ground_truth = dict()

    def observe(self, image: np.ndarray, particle: np.ndarray, point_index:int, **kwargs) -> None:
        """
            Calc the HSV histogramms for all the three channels
            :param image: part of image around particle
            :param particle: particle on image with pos[x][y]
            :param point_index: idx of point
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,self.channel_idx]
        maxVal = 256
        # max value of h-channel is 180, else 256 from docu
        if self.channel_idx == 0:
            maxVal = 180
        self.observation[point_index], _ = np.histogram(hsv.ravel(), bins=np.linspace(0, maxVal, self.bins + 1), range=[0,maxVal])
        if not (point_index in self.ground_truth.keys()):
            self.ground_truth[point_index] = self.observation[point_index].copy()

    def error(self, image: np.ndarray, particle: np.ndarray, point_index:int, **kwargs) -> float:
        """
            Calc the error for the HSV histogramms between the current image and the groundtruth
            :param image: hole image
            :param obervation: hsv histogram of particle
            :param particle: particle on image with pos[x][y]
            :param point_index: idx of point
            :param ground_truth: ground truth hsv histogram
        """
        dist = 0

        if np.sum(self.observation[point_index]) == 0.0:
            return self.max_error()

        if self.metrics == HIST_HSV_FEATURE.EUCLIDEAN_DIST:
            diff = self.observation[point_index] - self.ground_truth[point_index]
            dist = np.linalg.norm(diff) #calc euclidean distance
        else:
            observation_norm = self.observation[point_index].astype(np.float32)
            ground_truth_norm = self.ground_truth[point_index].astype(np.float32)
            # normalize the histograms for comparison
            cv2.normalize(observation_norm, observation_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(ground_truth_norm, ground_truth_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            dist = cv2.compareHist(ground_truth_norm, observation_norm, int(self.metrics)-1)
            # normalize the dist for the choosen metric -> small is best
            if self.metrics == HIST_HSV_FEATURE.CORRELATION:
                dist = 1 - dist
            if self.metrics == HIST_HSV_FEATURE.CHISQR:
                dist = dist
        # weight the HSV Feature with importance factor for global comparison with other features 
        return dist * self.importance

class HistogrammRGBFeature(ParticleFeature):
    """
        Histogram Feature based on RGB Colorspace for Particle Filter
    """
    max_err = 1.0
    def __init__(self, ground_truth, **kwargs) -> None:
        """
            init the HistogramRGBFeature
            :param ground_truth: histogram on first original image
            :param **kwargs: 'importance' to weight the feature impact; 'metric' choose the metric of histogram difference; 'channel_idx' select the channel of histogram; 'bins' number of histogram bins
        """
        super().__init__(ground_truth, kwargs.get("importance", 1.0))
        self.bins = kwargs.get("bins", 16)
        self.channel_idx = kwargs.get("channel_idx")
        self.metrics = kwargs.get("metric", HIST_HSV_FEATURE.EUCLIDEAN_DIST)
        self.observation = dict()
        self.ground_truth = dict()

    def observe(self, image: np.ndarray, particle: np.ndarray, point_index:int, **kwargs) -> None:
        """
            Calc the RGB histogramms for all the three channels
            :param image: part of image around particle
            :param particle: particle on image with pos[x][y]
            :param point_index: idx of point
        """

        self.observation[point_index], _ = np.histogram(image[:,:,self.channel_idx].ravel(), bins=np.linspace(0, 256, self.bins + 1), range=[0,256])
        if not (point_index in self.ground_truth.keys()):
            self.ground_truth[point_index] = self.observation[point_index].copy()

    def error(self, image: np.ndarray, particle: np.ndarray, point_index:int, **kwargs) -> float:
        """
            Calc the error for the RGB histogramms between the current image and the groundtruth
            :param image: hole image
            :param observation: hsv histogram of particle
            :param particle: particle on image with pos[x][y]
            :param point_index: idx of point
            :param ground_truth: ground truth hsv histogram
        """
        dist = 0

        if self.metrics == HIST_HSV_FEATURE.EUCLIDEAN_DIST:
            diff = self.observation[point_index] - self.ground_truth[point_index]
            dist = np.linalg.norm(diff) #calc euclidean distance
        else:
            observation_norm = self.observation[point_index].astype(np.float32)
            ground_truth_norm = self.ground_truth[point_index].astype(np.float32)
            # normalize the histograms for comparison
            cv2.normalize(observation_norm, observation_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(ground_truth_norm, ground_truth_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            dist = cv2.compareHist(ground_truth_norm, observation_norm, int(self.metrics)-1)
            # normalize the dist for the choosen metric -> small is best
            if self.metrics == HIST_HSV_FEATURE.CORRELATION:
                dist = 1 - dist
            if self.metrics == HIST_HSV_FEATURE.CHISQR:
                dist = dist

        if dist != dist:
            raise Exception("Dist was NaN!")
        # weight the RGB Feature with importance factor for global comparison with other features 
        return dist * self.importance

class DistanceFeature(ParticleFeature):
    """
        Distance Feature for Particle Filter
    """
    max_err = 1.0
    def __init__(self, ground_truth, **kwargs) -> None:
        """
            init the Distance Feature
            :param ground_truth: not used
            :param **kwargs: 'importance' to weight the feature impact; 'metric' choose the metric of histogram difference
        """
        super().__init__(ground_truth, kwargs.get("importance", 1.0))
        self.metrics = kwargs.get("metric", HIST_HSV_FEATURE.EUCLIDEAN_DIST)
        self.max_err = 1.0
        self.ground_truth = None
        self.max_dist = None

    def observe(self, image: np.ndarray, particle: np.ndarray, point_index:int, **kwargs) -> None:
        """
            Set the groundtruth, else return zero-matrix of positions
            :param image: part of image around particle
            :param particle: particle on image with pos[x][y]
            :param point_index: idx of point
        """
        positions = np.zeros((ParticleFeature.max_observation_channels), dtype=np.float)

        #for i in range(particle.shape[0]):
        #    positions[i * 2] = particle[i][0]
        #    positions[i * 2 + 1] = particle[i][1]

        if self.ground_truth is None:
            self.ground_truth = FeatureHelper.getDistanceMatrix(particle, particle)
            self.max_dist = self.ground_truth.max()

    def error(self, image: np.ndarray, particle: np.ndarray, point_index:int, **kwargs) -> float:
        """
            Calc the error of the distance feature
            :param image: hole image
            :param observation: zero-vector
            :param particle: particle on image with pos[x][y]
            :param point_index: idx of point
            :param ground_truth: ground truth start positions
        """
        if self.max_dist > 0.0:
            diff = 0.0
            for j in range(particle.shape[0]):
                diff = diff + abs(self.ground_truth[point_index][j] - np.linalg.norm(particle[point_index] - particle[j]))
            #return the relative movement of particles of current frame to ground truth
            return np.clip(diff / (particle.shape[0] * self.max_dist), 0.0, 1.0) * self.importance
        else:
            # not selected
            return self.importance
 
