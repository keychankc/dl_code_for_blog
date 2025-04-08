import multi_object_tracking
import multi_object_tracking_slow
import multi_object_tracking_fast

if __name__ == '__main__':
    # multi_object_tracking.tracking("videos/soccer_01.mp4", "medianflow")

    # multi_object_tracking_slow.tracking("videos/race.mp4",
    #                                     "mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
    #                                     "mobilenet_ssd/MobileNetSSD_deploy.prototxt",
    #                                     "output/race_slow.mp4",
    #                                     0.2)

    multi_object_tracking_fast.tracking("videos/race.mp4",
                                        "mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
                                        "mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                                        "output/race_fast.mp4",
                                        0.2)
