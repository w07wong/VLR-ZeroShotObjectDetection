import os
import shutil
import os.path as path
import numpy as np
import pickle as pkl
import sys
<<<<<<< HEAD
import csv

def process(scene_data_dir, target_images_dir, dataset_dir, video_folders, is_train=True):
   
    if not path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    if is_train:
        f = open(f'{dataset_dir}/train.csv', 'w') 
    else:
        f = open(f'{dataset_dir}/test.csv', 'w') 
        
    writer = csv.writer(f)
        
    video_frames_path = [path.join(scene_data_dir, vid) for vid in video_folders]
    cnt = 0
    
    class_list = []
    
    target_pose_total = 110
    target_pose_samples = 16
    frame_interval = 15

    for vid_path in video_frames_path:
        print(vid_path)
        for root,dir,files in os.walk(vid_path):
            for f in tqdm(files):
                if f.startswith('._') or not f.endswith('-color.png'):
                    continue
                id = f[:6]

                if int(id) % frame_interval != 0:
                    continue

                src_path = path.join(root, f)
                bbox_path = path.join(root, id+"-box.txt")
                mask_src_path = path.join(root, id+"-label.png")

                mask = cv2.imread(mask_src_path, cv2.IMREAD_GRAYSCALE)

                with open(bbox_path, 'r') as fin:
                    for obj_idx, line in enumerate(fin):
                        tmp = line.split()
                        class_label = tmp[0]
                        x1,y1,x2,y2 = ( float(n) for n in tmp[1:])
                        bbox = np.array([x1,y1,x2,y2])

                        # get mask
                        mask_id = class_id_map[class_label]
                        object_mask = np.zeros(mask.shape)
                        object_mask[np.where(mask==mask_id)] = 1

                        mask_src_path = path.join(dataset_dir, "%s_%d_mask.npy"%(id, obj_idx))
                        np.save(mask_src_path, object_mask)
                        
                        # get target images 
                        indices = np.arange(target_pose_total)
                        np.random.shuffle(indices)
                        #try:
                        #    assert np.where(mask==mask_id)[0].min()>=bbox[1]
                        #    assert np.where(mask==mask_id)[0].max()<=bbox[3]
                        #    assert np.where(mask==mask_id)[1].min()>=bbox[0]
                        #    assert np.where(mask==mask_id)[1].max()<=bbox[2]
                        #except:
                        #    from IPython import embed;embed()

                        target_src_dir = path.join(target_images_dir, class_label)
                        row = []
                        row.append(src_path)
                        row.extend(bbox.tolist())

                        for e,ind in enumerate(indices[:target_pose_samples]):
                            fname = "%04d.png"%ind
                            target_src_path = path.join(target_src_dir, fname)
                            target_image_name = "%06d_target.png"%(cnt)
                            target_dst_path = path.join(dataset_dir, target_image_name)
                            #os.symlink(target_src_path, target_dst_path)
                            row.append(target_src_path)

                            # move scene image
                            image_name = "%06d_scene.png"%cnt
                            dst_path = path.join(dataset_dir, image_name)
                            #os.symlink(src_path, dst_path)
                            # get bounding box (first item only)

                            bbox_path = path.join(dataset_dir, "%06d.npy"%cnt)
                            #np.save(bbox_path, bbox)
                            class_list.append((tmp[0]))

                            mask_dst_path = path.join(dataset_dir, "%06d_mask.npy"%cnt)
                            os.symlink(mask_src_path, mask_dst_path)

                            cnt += 1
                            #from IPython import embed;embed()

                        writer.writerow(row)

    #np.save(path.join(dataset_dir, "class.npy"), class_list)

def main():

    # CHANGE THIS: path to scene image folders
    scene_data_dir = "/home/rohanc/vlr/project/dataset/YCB/data/"
    # CHANGE THIS: full path to target image folder
    target_images_dir = "/home/rohanc/vlr/project/dataset/YCB/target/" #path.join(os.getcwd(), "target")
    # CHANGE THIS: path to dataset directory
    dataset_dir = "/home/rohanc/vlr/project/dataset/YCB/output/"
    # video index
    video_folders = ["0048","0049","0050"]

    if len(sys.argv) == 4:
        scene_data_dir = sys.argv[1]
        target_images_dir = sys.argv[2]
        dataset_dir = sys.argv[3]

    train_dataset_dir = path.join(dataset_dir, "train")
    test_dataset_dir = path.join(dataset_dir, "test")
    
    train_video_folders = ["0048", "0049"]
    test_video_folders = ["0050"]
    process(scene_data_dir, target_images_dir, train_dataset_dir, train_video_folders, is_train=True)
    process(scene_data_dir, target_images_dir, test_dataset_dir, test_video_folders, is_train=False)

    #from IPython import embed;embed()

if __name__ == "__main__":
    main()
