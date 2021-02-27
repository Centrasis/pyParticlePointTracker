import cv2
import numpy as np
from numpy.core.fromnumeric import shape
from ParticlePointTracker import ParticlePointTracker, Features

def gen_features():
    features = []
    #features.append(Features.HistogrammHSVFeature(None, channel_idx=0, metric=Features.HIST_HSV_FEATURE.BHATTACHARYYA, importance=1.0, bins=180))
    #features.append(Features.HistogrammHSVFeature(None, channel_idx=1, metric=Features.HIST_HSV_FEATURE.BHATTACHARYYA, importance=1.0, bins=128))
    #features.append(Features.HistogrammHSVFeature(None, channel_idx=2, metric=Features.HIST_HSV_FEATURE.BHATTACHARYYA, importance=1.0, bins=128))
    #features.append(Features.SIFTFeature(None))
    features.append(Features.TemplateFeature(None))
    #features.append(Features.GLCMFeature(None, importance=len(Features.GLCMFeature.measures)))
    return features

def test_integrity():
    ppt = ParticlePointTracker.ParticlePointTracker(gen_features, 40, 50, True)
    frame = np.zeros((500, 500), dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    points = np.array([[50, 50]], dtype=np.int)
    ppt.set_initial_points(points)
    ppt.step(frame)

    assert(ppt.resulting_points[0][0] == points[0][0] and ppt.resulting_points[0][1] == points[0][1])

def test_basic_tracking():
    ppt = ParticlePointTracker.ParticlePointTracker(gen_features, 40, 100, True)
    points = np.array([[50, 50]], dtype=np.int)
    ok_counter = 1
    for _ in range(10):
        ppt.__reset__()
        ppt.set_initial_points(points)
        for i in range(5):   
            offset = i * 5
            frame = np.zeros((200, 200), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = cv2.circle(frame, (50, 50 + offset), 10, (255, 50, 20), 2)
            ppt.step(frame)
            
            if ppt.resulting_points[0][1] > 45 + offset and ppt.resulting_points[0][1] < 45 + offset + 20:
                if ppt.resulting_points[0][0] > 30 and ppt.resulting_points[0][0] < 70:
                    ok_counter += 1
    
    assert(ok_counter > 5)