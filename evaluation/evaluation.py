import numpy as np
from math import cos, sin, atan2, asin
import os
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import sys
if sys.version > '3':
   import pickle as cPickle
else:
   import cPickle

def estimate_pose(points_static, points_to_transform):
    #pdb.set_trace()
    p0 = np.copy(points_static).T
    p1 = np.copy(points_to_transform).T

    t0 = -np.mean(p0, axis=1).reshape(3,1)
    t1 = -np.mean(p1, axis=1).reshape(3,1)
    

    p0c = p0+t0
    p1c = p1+t1

    covariance_matrix = p0c.dot(p1c.T)
    U,S,V = np.linalg.svd(covariance_matrix)
    R = U.dot(V)
    if np.linalg.det(R) < 0:
        R[:,2] *= -1

    rms_d0 = np.sqrt(np.mean(np.linalg.norm(p0c, axis=0)**2))
    rms_d1 = np.sqrt(np.mean(np.linalg.norm(p1c, axis=0)**2))

    s = (rms_d0/rms_d1)
    P = s*np.eye(3).dot(R)


    t_final = P.dot(t1) -t0
    P = np.c_[P, t_final]

    return P

def process(pre_rst_path, gt_labels_path):

    pose_align_idx = [52, 55, 58, 61, 46, 84, 90]
    ## load gt labels
    fid = open(gt_labels_path, 'rb')
    if sys.version > '3':
       gt_labels = cPickle.load(fid, encoding="latin1")
    else:
       gt_labels = cPickle.load(fid)
    fid.close() 
    #gt_labels = np.load(gt_labels_path)
    num = len(gt_labels)

    ## load pre labels
    with open(pre_rst_path,'r') as fid:
        lines = fid.readlines()
 
    ## computer errors
    full_errors = []
    inner_face_errors = []
    count = 0
    for i in range(num):
        idx = i*107
        name = lines[idx].strip()
        print(name)
        pre_points = []
        for j in range(106):
            line = lines[idx+j+1]
            s_str = line.split()
            pts = [float(x) for x in s_str]
            pre_points.append(pts[0:3])
        pre_points = np.array(pre_points)

        ## pose alignment
        gt_points = gt_labels[name]
        Proj = estimate_pose(gt_points[pose_align_idx, :], pre_points[pose_align_idx,:])
        pre_points = np.hstack((pre_points, np.ones((pre_points.shape[0], 1))))
        aligned_pre_points = Proj.dot(pre_points.T)
        aligned_pre_points = aligned_pre_points.T
        ## error
        #pdb.set_trace()
        errors = np.linalg.norm(aligned_pre_points- gt_points, axis=1)
        interocular_distance = np.linalg.norm(gt_points[74]- gt_points[77])
        full_errors.append(np.sum(errors)/(106*interocular_distance))
        inner_face_errors.append(np.sum(errors[33:])/(73*interocular_distance))

    return full_errors, inner_face_errors

def get_files(rootDir):
    list_dirs = os.walk(rootDir)
    file_lists = []
    #pdb.set_trace()
    for root, dirs, files in list_dirs:
        for f in files:
            file_lists.append(os.path.join(root, f))
    return file_lists

def drawROC( data_path, nme_x=0.1):
    
    lists = get_files(data_path)
    npy_lists = [name for name in lists
            if name.endswith('npy')]
    print("total methods: %d\n"%(len(npy_lists)))
   
    num_methods = len(npy_lists)

    mean_error_array = []
    align_errors_array = []
    name_array = []
    AUC_array = []
    Failure_rate_array = []

    for npy_file in npy_lists:
        roc_name = npy_file.split('/')[-1]
        roc_name = roc_name.split('.')[0]
        align_errors = np.load(npy_file)
        mean_error = np.mean(align_errors)
        mean_error_array.append(mean_error)
        align_errors_array.append(align_errors)
        name_array.append(roc_name)

        align_errors_sort = sorted(align_errors)
        align_errors_sort.append(1)
        num = len(align_errors_sort)
        tmp = range(1,num)
        p = [elem*100.0/num for elem in tmp]
        p.append(100)
        ######## AUC ###########
        error_bins = []
        acc_bins = []
        for i in range(num-1):
            error_bins.append(align_errors_sort[i+1] - align_errors_sort[i])
            acc_bins.append(error_bins[i]*0.5*(p[i]+p[i+1]))
        #acc_bins = np.array(acc_bins)
        mm = abs(np.array(align_errors_sort) - nme_x)
        idx_array = np.where(mm==np.min(mm))
        idx = idx_array[0][0]
        AUC = 0
        if (nme_x - align_errors_sort[idx])>0:
           AUC = np.sum(acc_bins[0:idx]) + (nme_x - align_errors_sort[idx])*acc_bins[idx]/error_bins[idx]
        else:
           AUC = np.sum(acc_bins[0:idx]) + (nme_x - align_errors_sort[idx])*acc_bins[idx-1]/error_bins[idx-1]
        AUC = AUC/(100*nme_x)
        AUC_array.append(AUC)
        Failure_rate_array.append(100-p[idx])
        print("%s  Mean error: %f  AUC(@%0.2f): %f Failure rate: %f\n" %(roc_name, mean_error, nme_x, AUC, 100-p[idx]))

    ##### only show top 5 
    plt.figure()
    plt.figure(figsize=(8,8))
    color_array = ['r', 'g', 'b', 'k', 'y', 'c', 'm']

    sorted_idx = np.argsort(mean_error_array)
    num = 5 if num_methods>5 else num_methods
    for ii in range(num):
        ind = sorted_idx[ii]
        align_errors = align_errors_array[ind]
        roc_name = name_array[ind]
        mean_error = mean_error_array[ind]

        align_errors_sort = sorted(align_errors)
        align_errors_sort.append(1)
        num = len(align_errors_sort)
        tmp = range(1,num)
        p = [elem*100.0/num for elem in tmp]
        p.append(100)
        ######## AUC ###########
        error_bins = []
        acc_bins = []
        for i in range(num-1):
            error_bins.append(align_errors_sort[i+1] - align_errors_sort[i])
            acc_bins.append(error_bins[i]*0.5*(p[i]+p[i+1]))
        #acc_bins = np.array(acc_bins)
        mm = abs(np.array(align_errors_sort) - nme_x)
        idx_array = np.where(mm==np.min(mm))
        idx = idx_array[0][0]
        AUC = 0
        if (nme_x - align_errors_sort[idx])>0:
           AUC = np.sum(acc_bins[0:idx]) + (nme_x - align_errors_sort[idx])*acc_bins[idx]/error_bins[idx]
        else:
           AUC = np.sum(acc_bins[0:idx]) + (nme_x - align_errors_sort[idx])*acc_bins[idx-1]/error_bins[idx-1]
        AUC = AUC/(100*nme_x)
        
        ####### draw roc curve#######
        plt.plot(align_errors_sort, p, color=color_array[ii], lw=2, label=roc_name+'(nme:%0.4f)'%mean_error)
        

    #################
    plt.xlim([0.0, nme_x])
    plt.ylim([0.0, 100])
    plt.xlabel('Error metric')
    plt.ylabel('Cumulative correct rate')
    plt.grid(color='k', linestyle='--', linewidth=1)
    plt.title('Performance in testing phase')
    plt.legend(loc="lower right")
    plt.savefig('ROC.png')
    plt.show()

def saveData(name,data):
    fileObject = open(name, 'w')
    for ip in data:
        fileObject.write('%f\n'%(ip))
    fileObject.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evalation')
    parser.add_argument('--pre_rst_path', default='val_baseline.txt',
                        help='predict result path', type=str)
    parser.add_argument('--gt_label_path', default='val_gt.pkl', type=str, help='ground truth result path')
    args = parser.parse_args()

    try:
       full_errors, inner_face_errors = process(args.pre_rst_path, args.gt_label_path)
       name = args.pre_rst_path.split('/')[-1]
       name = name.split('.')[0]
       np.save("methods_rst/{}.npy".format(name), full_errors)
       drawROC('methods_rst')
       print('Success!')
    except:
       print('Input data format error!')    


