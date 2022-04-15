import os
import shutil
import os.path as path
import numpy as np
import pickle as pkl


# CHANGE THIS: path to scene image folders
data_dir = "/Volumes/wenyu/16824-VLR/YCB/YCB_dataset/data"
# CHANGE THIS: path to target image folder
target_images_dir = "ycb_models"
# video index
video_folders = ["0048","0049","0050"]


video_frames_path = [path.join(data_dir, vid) for vid in video_folders]
dataset_dir = "datasets"

def main():
    cnt = 0
    if not path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    
    class_list = []

    for vid_path in video_frames_path:
        print(vid_path)
        for root,dir,files in os.walk(vid_path):
            for f in files:
                if not f.endswith('-color.png'):
                    continue
                id = f[:6]
                # move scene image
                src_path = path.join(root, f)
                image_name = "%06d_scene.png"%cnt
                dst_path = path.join(dataset_dir, image_name)
                shutil.copyfile(src_path, dst_path)
                # get bounding box (first item only)
                bbox_path = path.join(root, id+"-box.txt")
                with open(bbox_path, 'r') as fin:
                    for line in fin:
                        tmp = line.split()
                        class_label = tmp[0]
                        x1,y1,x2,y2 = ( float(n) for n in tmp[1:])
                        bbox = np.array([x1,y1,x2,y2])
                        bbox_path = path.join(dataset_dir, "%06d.npy"%cnt)
                        np.save(bbox_path, bbox)
                        class_list.append((tmp[0]))
                        break
                # get target image (TODO: a set of target images with different poses)
                target_src_path = path.join(target_images_dir, class_label+".png")
                target_image_name = "%06d_target.png"%cnt
                target_dst_path = path.join(dataset_dir, target_image_name)
                shutil.copyfile(target_src_path, target_dst_path)
                cnt += 1

    np.save(path.join(dataset_dir, "class.npy"), class_list)

    #from IPython import embed;embed()

if __name__ == "__main__":
    main()
