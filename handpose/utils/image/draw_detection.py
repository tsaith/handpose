from . import draw_bbox


def draw_detection(image, image_h, image_w, detections, classes, colors):
    # write result images. Draw bounding boxes and labels of detections

    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

        x1 = x1.item()
        y1 = y1.item()
        x2 = x2.item()
        y2 = y2.item()

        # Rescale coordinates to original dimensions
        ori_w, ori_h = image.size
        pre_w, pre_h = image_w, image_h
        bbox_h = ((y2 - y1) / pre_h) * ori_h
        bbox_w = ((x2 - x1) / pre_w) * ori_w
        y1 = (y1 / pre_h) * ori_h
        x1 = (x1 / pre_w) * ori_w

        # Draw the bbox
        bbox = (x1, y1, x1+bbox_w, y1+bbox_h)
        cls_index = int(cls_pred)
        lb = "{}({:4.2f})".format(classes[cls_index], cls_conf)
        draw_bbox(image, bbox, label=lb, color=colors[cls_index])

    return bbox

