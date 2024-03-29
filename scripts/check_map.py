import numpy as np
from mean_average_precision import MetricBuilder


def main():
    # [xmin, ymin, xmax, ymax, class_id, difficult]
    gt = np.array([
        [439, 157, 556, 241, 0, 0],
        [437, 246, 518, 351, 0, 0],
        [515, 306, 595, 375, 0, 0],
        [407, 386, 531, 476, 0, 0],
        [544, 419, 621, 476, 0, 0],
        [609, 297, 636, 392, 0, 0],
    ])

    # [xmin, ymin, xmax, ymax, class_id, confidence]
    preds = np.array([
        [429, 219, 528, 247, 0, 0.460851],
        [433, 260, 506, 336, 0, 0.269833],
        [518, 314, 603, 369, 0, 0.462608],
        [592, 310, 634, 388, 0, 0.298196],
        [403, 384, 517, 461, 0, 0.382881],
        [405, 429, 519, 470, 0, 0.369369],
        [433, 272, 499, 341, 0, 0.272826],
        [413, 390, 515, 459, 0, 0.619459],
    ])

    # print list of available metrics
    print(MetricBuilder.get_metrics_list())

    # create metric_fn
    metric_fn = MetricBuilder.build_evaluation_metric(
        "map_2d", async_mode=True, num_classes=1)

    # add some samples to evaluation
    for i in range(10):
        metric_fn.add(preds, gt)

    metrics = metric_fn.value(
        iou_thresholds=np.arange(0.5, 1.0, 0.05),
        recall_thresholds=np.arange(0., 1.01, 0.01),
        mpolicy='soft')['mAP']

    print(f"COCO mAP: {metrics['mAP']}")


if __name__ == '__main__':
    main()
