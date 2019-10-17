# import csv
# import os

# import numpy as np

# from datasets.video_stream import VideoStream


# def causal_map_func(submission_array, gt_array, w_array):
#     """ Returns mAP, weighted mAP, and AP array """
#     m_aps = []
#     n_classes = submission_array.shape[1]
#     for oc_i in range(n_classes):
#         sorted_idxs = np.argsort(-submission_array[:, oc_i])
#         tp = gt_array[:, oc_i][sorted_idxs] == 1
#         fp = np.invert(tp)
#         n_pos = tp.sum()

#         t_pcs = np.cumsum(tp)
#         f_pcs = np.cumsum(fp)

#         w_t_pcs = t_pcs * w_array[oc_i]
#         prec = w_t_pcs / (f_pcs + w_t_pcs).astype(float)
#         avg_prec = 0
#         for i in range(submission_array.shape[0]):
#             if tp[i]:
#                 avg_prec += prec[i]
#         m_aps.append(avg_prec / n_pos.astype(float))
#     m_aps = np.array(m_aps)
#     m_ap = np.nanmean(m_aps)
#     w_ap = sum(m_aps * gt_array.sum(axis=0) / gt_array.sum().astype(float))
#     return m_ap, w_ap, m_aps


# def causal_map(submission_array, gt_array, w_array):
#     fix = submission_array.copy()
#     empty = np.sum(gt_array, axis=1) == 0
#     fix[empty, :] = np.NINF

#     return causal_map_func(fix, gt_array, w_array)


# def load_causal_charades(gt_path, data_path):
#     gt_ids = []
#     gt_classes = []
#     with open(gt_path) as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             vid = row['id']
#             actions = row['actions']
#             if actions == '':
#                 actions = []
#             else:
#                 actions = [a.split(' ') for a in actions.split(';')]
#                 actions = [{
#                     'class': x,
#                     'start': float(y),
#                     'end': float(z)
#                 } for x, y, z in actions]
#             video_stream = VideoStream(os.path.join(data_path, vid), actions)
#             first_frame = video_stream.dataset.first_frame
#             frame_target = video_stream.dataset.target['target']
#             for f_id, frame_t in enumerate(frame_target):
#                 gt_ids.append('{}_{:06d}'.format(vid, first_frame + f_id))
#                 gt_classes.append(frame_t)

#     return gt_ids, gt_classes
