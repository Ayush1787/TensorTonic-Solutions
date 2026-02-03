import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    pooled_outputs = []
    H = len(feature_map)
    W = len(feature_map[0])

    for roi in rois:
        x1, y1, x2, y2 = roi
        roi_h = y2 - y1
        roi_w = x2 - x1

        pooled = [[0 for _ in range(output_size)] for _ in range(output_size)]

        for i in range(output_size):          # height bins
            for j in range(output_size):      # width bins

                # Compute bin boundaries using floor
                h_start = y1 + (i * roi_h) // output_size
                h_end   = y1 + ((i + 1) * roi_h) // output_size
                w_start = x1 + (j * roi_w) // output_size
                w_end   = x1 + ((j + 1) * roi_w) // output_size

                # Ensure at least one pixel per bin
                if h_end == h_start:
                    h_end = min(h_start + 1, H)
                if w_end == w_start:
                    w_end = min(w_start + 1, W)

                # Max pooling
                max_val = float('-inf')
                for y in range(h_start, h_end):
                    for x in range(w_start, w_end):
                        max_val = max(max_val, feature_map[y][x])

                pooled[i][j] = max_val

        pooled_outputs.append(pooled)

    return pooled_outputs
