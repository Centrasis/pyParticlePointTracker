import cv2
import numpy as np
from numpy.core.fromnumeric import shape
from ParticlePointTracker import ParticlePointTracker, Features

def gen_features():
    features = []
    features.append(Features.HistogrammHSVFeature(None, channel_idx=0, metric=Features.HIST_HSV_FEATURE.BHATTACHARYYA, importance=1.0, bins=180))
    features.append(Features.HistogrammHSVFeature(None, channel_idx=1, metric=Features.HIST_HSV_FEATURE.BHATTACHARYYA, importance=1.0, bins=128))
    features.append(Features.HistogrammHSVFeature(None, channel_idx=2, metric=Features.HIST_HSV_FEATURE.BHATTACHARYYA, importance=1.0, bins=128))
    features.append(Features.SIFTFeature(None))
    features.append(Features.TemplateFeature(None))
    features.append(Features.GLCMFeature(None, importance=len(Features.GLCMFeature.measures)))
    return features

def test_integrity():
    ppt = ParticlePointTracker.ParticlePointTracker(gen_features, 40, 50, True)
    frame = np.zeros((500, 500), dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    frame = cv2.circle(frame, (50, 50), 10, (255, 50, 20), 2)
    points = np.array([[50, 50]], dtype=np.int)
    ppt.set_initial_points(points)
    ppt.step(frame)

    assert(ppt.resulting_points[0][0] == points[0][0] and ppt.resulting_points[0][1] == points[0][1])

    frame = np.zeros((500, 500), dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    frame = cv2.circle(frame, (50, 60), 10, (255, 50, 20), 2)
    ppt.step(frame)

    assert(ppt.resulting_points[0][1] > 50 and ppt.resulting_points[0][1] < 70)
    assert(ppt.resulting_points[0][0] > 45 and ppt.resulting_points[0][0] < 55)