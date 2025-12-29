import json

def evaluate_customised_dataset(ground_truth_json, pred_json_file, confidence_threshold=0.5):
    # Read ground truth and prediction JSON files
    with open(ground_truth_json, "r") as f:
        ground_truth = json.load(f)

    with open(pred_json_file, "r") as f:
        predictions = json.load(f)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for prediction in predictions:
        image_id = prediction["image_id"]
        category_id = prediction["category_id"]
        bbox = prediction["bbox"]
        score = prediction["score"]

        # Find ground truth associated with the prediction
        matching_ground_truth = [
            gt for gt in ground_truth if gt["image_id"] == image_id and gt["category_id"] == category_id
        ]

        if not matching_ground_truth:
            # If there are no matching ground truths, consider the prediction a false positive
            false_positives += 1
        else:
            # If there are matching ground truths, find the ground truth with the highest IOU
            best_iou = 0
            best_match = None

            for gt in matching_ground_truth:
                gt_bbox = gt["bbox"]

                # Calculate IOU
                x1 = max(bbox[0], gt_bbox[0])
                y1 = max(bbox[1], gt_bbox[1])
                x2 = min(bbox[0] + bbox[2], gt_bbox[0] + gt_bbox[2])
                y2 = min(bbox[1] + bbox[3], gt_bbox[1] + gt_bbox[3])

                intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
                prediction_area = bbox[2] * bbox[3]
                ground_truth_area = gt_bbox[2] * gt_bbox[3]
                iou = intersection_area / (prediction_area + ground_truth_area - intersection_area)

                if iou > best_iou:
                    best_iou = iou
                    best_match = gt

            if best_iou >= confidence_threshold:
                # If the best matching IOU is above the threshold, consider the prediction a true positive
                true_positives += 1
                # Remove the matched ground truth from the list to avoid multiple matches
                ground_truth.remove(best_match)
            else:
                # Otherwise, consider the prediction a false positive
                false_positives += 1

    # Calculate unmatched predictions, which are predictions that were not correct matches with ground truth
    unmatched_predictions = max(0, len(ground_truth) - true_positives)
    false_negatives += unmatched_predictions

    # Calculate precision and recall metrics
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    precision_percent = "{:.2%}".format(precision)
    recall_percent = "{:.2%}".format(recall)

    print("Precision:", precision_percent)
    print("Recall:", recall_percent)



if __name__ == "__main__":
    ground_truth_json = 'instances_val2017.json'
    pred_json_file = 'best_bs4_6.0_predictions.json'
    evaluate_customised_dataset(ground_truth_json, pred_json_file)