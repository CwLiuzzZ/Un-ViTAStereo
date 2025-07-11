import numpy as np
import cv2
# import sys
# sys.path.append('..')
# sys.path.append('../..')
from toolkit.function.base_function import io_disp_read
from toolkit.data_loader.dataset_function import generate_file_lists

def generate_appended_file_dirs(dataset,len_):
    disp_list_noc = np.zeros((len_))
    disp_list_fd = np.zeros((len_))
    if dataset in ['realroad','realroad_pt']:
        disp_list_noc = generate_file_lists(dataset = dataset,if_train=True,method='BGNet')['disp_list']
    elif dataset in ['realroad2','realroad2_pt']:
        disp_list_noc = generate_file_lists(dataset = dataset,if_train=True,method='noc_mask')['disp_list']
    elif 'VirtualRoad' in dataset:
        disp_list_fd = generate_file_lists(dataset = dataset,if_train=True,method='pt_disp')['disp_list']
        disp_list_noc = generate_file_lists(dataset = 'VirtualRoad',if_train=True,method='gt')['disp_list']
    return disp_list_noc,disp_list_fd


# post process the disp result, such as mask the noc area, and add the pt fd disp to recover
def results_post_process(result,left_dir,dataset,dataset_type,data,noc_dir=None,pt_fd_dir=None):
    ori_img = cv2.imread(left_dir)
    ori_H,ori_W = ori_img.shape[0],ori_img.shape[1]

    if 'middlebury' in dataset:
        if 'im0' in data['left_dir'][0]:
            noc_mask = cv2.imread(data['left_dir'][0].replace('im0.png','noc_mask.png'),-1)
        elif 'view1' in data['left_dir'][0]:
            noc_mask = cv2.imread(data['left_dir'][0].replace('view1.png','noc_mask.png'),-1)
        noc_mask = cv2.resize(noc_mask,(result.shape[1],result.shape[0]),interpolation=cv2.INTER_NEAREST)
        result = result*noc_mask                
    elif 'MiddEval3' in dataset and dataset_type == 'train':
        noc_mask = cv2.imread(data['left_dir'][0].replace('im0.png','noc_mask.png'),-1)
        noc_mask = cv2.resize(noc_mask,(result.shape[1],result.shape[0]),interpolation=cv2.INTER_NEAREST)
        result = result*noc_mask  
    elif 'realroad' in dataset:
        if "pt" in dataset:
            result = result+io_disp_read(data['disp_dir'][0])
        # print('mask the disp with {}'.format(noc_dir))
        mask_disp = io_disp_read(noc_dir) 
        result[mask_disp==0]=0
    elif 'VirtualRoad' in dataset:
        assert noc_dir.split('/')[-1].split('.')[0] == data['save_dir_disp'][0].split('/')[-1].split('.')[0] == pt_fd_dir.split('/')[-1].split('.')[0]
        if 'pt' in dataset:
            pt = io_disp_read(pt_fd_dir)
            # avoid adding pt_fd disp for sparse matching
            pt[result==0]=0
            result = result+pt
        # print('mask the disp with {}'.format(noc_dir))
        if result.shape[0] == ori_H and result.shape[1] == ori_W:
            gt_disp = io_disp_read(noc_dir)
            result[gt_disp==0]=0
        # # pure is fucking stupid
        # if 'pure' in data['left_dir'][0]:
        #     road_mask = data['left_dir'][0].replace('rgb_front_left','road_seg_mask').replace('pure_road_left','road_seg_mask')
        #     road_mask = cv2.imread(road_mask,-1)
        #     result[road_mask==0]=0
    return result
    










# # post process the disp result, such as mask the noc area, and add the pt fd disp to recover
# def results_post_process_evaluate(result,dataset,dataset_type,data,ori_W,ori_H,noc_dir=None,pt_fd_dir=None):

#     if 'middlebury' in dataset:
#         if 'im0' in data['left_dir'][0]:
#             noc_mask = cv2.imread(data['left_dir'][0].replace('im0.png','noc_mask.png'),-1)
#         elif 'view1' in data['left_dir'][0]:
#             noc_mask = cv2.imread(data['left_dir'][0].replace('view1.png','noc_mask.png'),-1)
#         noc_mask = cv2.resize(noc_mask,(result.shape[1],result.shape[0]),interpolation=cv2.INTER_NEAREST)
#         result = result*noc_mask
#     elif 'MiddEval3' in dataset and dataset_type == 'train':
#         noc_mask = cv2.imread(data['left_dir'][0].replace('im0.png','noc_mask.png'),-1)
#         noc_mask = cv2.resize(noc_mask,(result.shape[1],result.shape[0]),interpolation=cv2.INTER_NEAREST)
#         result = result*noc_mask  
#     elif 'realroad' in dataset:
#         if "pt" in dataset:
#             result = result+data['disp'].squeeze().detach().cpu().numpy()
#         # print('mask the disp with {}'.format(noc_dir))
#         mask_disp = io_disp_read(noc_dir)
#         result[mask_disp==0]=0

#     elif 'VirtualRoad' in dataset:
#         assert noc_dir.split('/')[-1].split('.')[0] == data['save_dir_disp'][0].split('/')[-1].split('.')[0] == pt_fd_dir['save_dir_disp'][0].split('/')[-1].split('.')[0]
#         if 'pt' in dataset:
#             # print(noc_dir, pt_fd_dir, data['save_dir_disp'][0])
#             assert noc_dir.split('/')[-1].split('.')[0] == pt_fd_dir.split('/')[-1].split('.')[0] == data['save_dir_disp'][0].split('/')[-1].split('.')[0]
#             result = result+io_disp_read(pt_fd_dir)
#         # print('mask the disp with {}'.format(data['disp_dir'][0]))
#         gt_disp = io_disp_read(noc_dir)
#         result[gt_disp==0]=0
#         # if 'pure' in data['left_dir'][0]:
#         #     road_mask = data['left_dir'][0].replace('rgb_front_left','road_seg_mask').replace('pure_road_left','road_seg_mask')
#         #     road_mask = cv2.imread(road_mask,-1)
#         #     result[road_mask==0]=0
#     return result





#########################################
#########################################
# if 'middlebury' in dataset:
#     if 'im0' in data['left_dir'][0]:
#         noc_mask = cv2.imread(data['left_dir'][0].replace('im0.png','noc_mask.png'),-1)
#     elif 'view1' in data['left_dir'][0]:
#         noc_mask = cv2.imread(data['left_dir'][0].replace('view1.png','noc_mask.png'),-1)
#     noc_mask = cv2.resize(noc_mask,(results.shape[1],results.shape[0]),interpolation=cv2.INTER_NEAREST)
#     results = results*noc_mask                
# elif 'MiddEval3' in dataset and dataset_type == 'train':
#     noc_mask = cv2.imread(data['left_dir'][0].replace('im0.png','noc_mask.png'),-1)
#     noc_mask = cv2.resize(noc_mask,(results.shape[1],results.shape[0]),interpolation=cv2.INTER_NEAREST)
#     results = results*noc_mask  
# elif 'realroad' in dataset:
#     if "pt" in dataset:
#         results = results+io_disp_read(data['disp_dir'][0])
#     # print('mask the disp with {}'.format(disp_list_mask[i]))
#     mask_disp = io_disp_read(disp_list_mask[i])
#     results[mask_disp==0]=0
# elif 'VirtualRoad' in dataset:
#     # print('mask the disp with {}'.format(disp_list_gt_noc[i]))
#     if 'pt' in dataset:
#         assert disp_list_gt_noc[i].split('/')[-1].split('.')[0] == disp_list_fd[i].split('/')[-1].split('.')[0] == data['save_dir_disp'][0].split('/')[-1].split('.')[0]
#         pt = io_disp_read(disp_list_fd[i])
#         pt[results==0]=0
#         results = results+pt
#     else:
#         assert disp_list_gt_noc[i].split('/')[-1].split('.')[0] == data['save_dir_disp'][0].split('/')[-1].split('.')[0]
#     if results.shape[0] == img_H and results.shape[1] == img_W:
#         gt_disp = io_disp_read(disp_list_gt_noc[i])
#         results[gt_disp==0]=0
#     if 'pure' in data['left_dir'][0]:
#         road_mask = data['left_dir'][0].replace('rgb_front_left','road_seg_mask').replace('pure_road_left','road_seg_mask')
#         road_mask = cv2.imread(road_mask,-1)
#         results[road_mask==0]=0
#########################################
#########################################
# if 'middlebury' in dataset:
#     if 'im0' in data['left_dir'][0]:
#         noc_mask = cv2.imread(data['left_dir'][0].replace('im0.png','noc_mask.png'),-1)
#     elif 'view1' in data['left_dir'][0]:
#         noc_mask = cv2.imread(data['left_dir'][0].replace('view1.png','noc_mask.png'),-1)
#     noc_mask = cv2.resize(noc_mask,(pred_disp.shape[1],pred_disp.shape[0]),interpolation=cv2.INTER_NEAREST)
#     pred_disp = pred_disp*noc_mask
# elif 'MiddEval3' in dataset and dataset_type == 'train':
#     noc_mask = cv2.imread(data['left_dir'][0].replace('im0.png','noc_mask.png'),-1)
#     noc_mask = cv2.resize(noc_mask,(pred_disp.shape[1],pred_disp.shape[0]),interpolation=cv2.INTER_NEAREST)
#     pred_disp = pred_disp*noc_mask  
# elif 'realroad' in dataset:
#     if "pt" in dataset:
#         pred_disp = pred_disp+data['disp'].squeeze().detach().cpu().numpy()
#     # print('mask the disp with {}'.format(disp_list_mask[i]))
#     mask_disp = io_disp_read(disp_list_mask[i])
#     pred_disp[mask_disp==0]=0

# elif 'VirtualRoad' in dataset:
#     if 'pt' in dataset:
#         # print(disp_list_gt_noc[i], disp_list_fd[i], data['save_dir_disp'][0])
#         assert disp_list_gt_noc[i].split('/')[-1].split('.')[0] == disp_list_fd[i].split('/')[-1].split('.')[0] == data['save_dir_disp'][0].split('/')[-1].split('.')[0]
#         pred_disp = pred_disp+io_disp_read(disp_list_fd[i])
#     else:
#         assert disp_list_gt_noc[i].split('/')[-1].split('.')[0] == data['save_dir_disp'][0].split('/')[-1].split('.')[0]
#     # print('mask the disp with {}'.format(data['disp_dir'][0]))
#     gt_disp = io_disp_read(disp_list_gt_noc[i])
#     pred_disp[gt_disp==0]=0
#     if 'pure' in data['left_dir'][0]:
#         road_mask = data['left_dir'][0].replace('rgb_front_left','road_seg_mask').replace('pure_road_left','road_seg_mask')
#         road_mask = cv2.imread(road_mask,-1)
#         pred_disp[road_mask==0]=0
#########################################
#########################################