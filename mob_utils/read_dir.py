import os

class ReadDir():
    def __init__(self,
                 base_dir,
                 subset='training',
                 tracklet_date='2011_09_26',
                 tracklet_file ='2011_09_26_drive_0084_sync'
                 ):
        # Todo: set local base dir
        self.base_dir = 'D:/Meus documentos/Desktop/3d-bounding-box-estimation-for-autonomous-driving-master'
        # self.base_dir = base_dir

        # if use kitti training data for train/val evaluation
        if subset == 'training':
            self.label_dir = os.path.join(self.base_dir, subset, 'label_2/')
            self.image_dir = os.path.join(self.base_dir, subset, 'image_2/')
            self.calib_dir = os.path.join(self.base_dir, subset, 'calib/')
            self.prediction_dir = os.path.join(self.base_dir, subset, 'box_3d/')

        # if use raw data
        if subset == 'testing':
            self.tracklet_drive = os.path.join(self.base_dir, tracklet_date, tracklet_file)
            self.label_dir = os.path.join(self.tracklet_drive, 'label_02/')
            self.image_dir = os.path.join(self.tracklet_drive, 'image_02/')
            self.calib_dir = os.path.join(self.tracklet_drive, 'calib_02/')
            self.prediction_dir = os.path.join(self.tracklet_drive, 'box_3d_mobilenet/')


if __name__ == '__main__':
    dir = ReadDir(subset='training')
    dir_ = ReadDir(subset='testing',
                    tracklet_date='2011_09_06',
                    tracklet_file='2011_09_26_drive_0084_sync')
    print(dir.image_dir)
    print(dir_.image_dir)
