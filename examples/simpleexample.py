import cv2
import numpy as np
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

if __name__ == "__main__":
    ppt = ParticlePointTracker.ParticlePointTracker(gen_features, 20, 50, True)
    video = cv2.VideoCapture(0)
    ret = True
    points = np.array([[200, 200], [100, 200], [300, 300]], dtype=np.int)
    ppt.set_initial_points(points)
    while ret:
        ret, frame = video.read()
        if not ret:
            continue

        ppt.step(frame)

        if ppt.resulting_points is None:
            continue

        new_points = ppt.resulting_points
        for i, p in enumerate(new_points):
            frame = cv2.circle(frame, (p[0], p[1]), 20, (255, 0, 0), 1)
        cv2.imshow("result", frame)
        cv2.waitKeyEx(1)