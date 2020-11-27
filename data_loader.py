import torch
import os
import datetime
import numpy as np
import cv2
from torch.utils import data
from torchvision import utils
import scipy.io
import os,glob
from torchvision import transforms
import PIL
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def preprocess_fixmaps(fix_map, shape_r, shape_c):
    ims = padding_fixation(fix_map, shape_r=shape_r, shape_c=shape_c)

    return ims

def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols),dtype=np.uint8)
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 255

    return out

def padding_fixation(img, shape_r=480, shape_c=640):
    img_padded = np.zeros((shape_r, shape_c))

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = resize_fixation(img, rows=shape_r, cols=new_cols)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols),] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_c)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded








def get_train_transforms(height=192, width=256):
    return A.Compose(
        [
            A.Resize(height=height, width=width, p=1.0),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ],
    )

def get_val_transforms(height=192, width=256):
    return A.Compose(
        [
            A.Resize(height=height, width=width, p=1.0),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ],
    )


class SALICONDataset(data.Dataset):

    source = 'SALICON'
    dynamic = False

    def __init__(self, phase='train', subset=None, verbose=1,
                 out_size=(288, 384), target_size=(480, 640),
                 preproc_cfg=None):
        self.resize = A.Resize(height=448, width=640, p=1.0)
        self.phase = phase
        if self.phase=='train':
            self.transforms=get_train_transforms()
        else:
            self.transforms=get_val_transforms()
        self.train = phase == 'train'
        self.subset = subset
        self.verbose = verbose
        self.out_size = out_size
        self.target_size = target_size
        self.preproc_cfg = {
            'rgb_mean': (0.485, 0.456, 0.406),
            'rgb_std': (0.229, 0.224, 0.225),
        }
        if preproc_cfg is not None:
            self.preproc_cfg.update(preproc_cfg)
        self.phase_str = 'val' if phase in ('valid', 'eval') else phase
        self.file_stem = f"COCO_{self.phase_str}2014_"
        self.file_nr = "{:012d}"

        self.samples = self.prepare_samples()
        if self.subset is not None:
            self.samples = self.samples[:int(len(self.samples) * subset)]
        # For compatibility with video datasets
        self.n_images_dict = {img_nr: 1 for img_nr in self.samples}
        self.target_size_dict = {
            img_nr: self.target_size for img_nr in self.samples}
        self.n_samples = len(self.samples)
        self.frame_modulo = 1

    def get_map(self, img_nr):
        map_file = self.dir + 'maps' +os.sep+ self.phase+os.sep+self.file_stem + self.file_nr.format(img_nr) + '.png'
        map = cv2.imread(str(map_file), cv2.IMREAD_GRAYSCALE)
        map = (map - np.min(map)) / (np.max(map) - np.min(map))
        assert(map is not None)
        return map

    def get_img(self, img_nr):
        img_file = self.dir + 'images' +os.sep+ self.phase+os.sep+self.file_stem + self.file_nr.format(img_nr) + '.jpg'

        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        # salbce
        # X = cv2.imread(str(img_file))
        # X = cv2.resize(X, (256,192), interpolation=cv2.INTER_AREA)
        # X = X.astype(np.float32)
        # X -= [103.939, 116.779, 123.68]
        # X = torch.FloatTensor(X)
        # X = X.permute(2, 0, 1)  # swap channel dimensions
        # img=X

        assert(img is not None)
        return img

    def get_raw_fixations(self, img_nr):
        raw_fix_file = self.dir + 'fixations' +os.sep+ self.phase+os.sep+self.file_stem + self.file_nr.format(img_nr) + '.mat'
        fix_data = scipy.io.loadmat(raw_fix_file)
        fixations_array = [gaze[2] for gaze in fix_data['gaze'][:, 0]]
        return fixations_array, fix_data['resolution'].tolist()[0]

    def process_raw_fixations(self, fixations_array, res):
        fix_map = np.zeros(res, dtype=np.uint8)
        for subject_fixations in fixations_array:
            fix_map[subject_fixations[:, 1] - 1, subject_fixations[:, 0] - 1]\
                = 255
        return fix_map



    def get_fixation_map(self, img_nr):
        fix_map_file = self.dir / 'fixations' / self.phase_str / (
                self.file_stem + self.file_nr.format(img_nr) + '.png')
        if fix_map_file.exists():
            fix_map = cv2.imread(str(fix_map_file), cv2.IMREAD_GRAYSCALE)
        else:
            fixations_array, res = self.get_raw_fixations(img_nr)
            fix_map = self.process_raw_fixations(fixations_array, res)
            cv2.imwrite(str(fix_map_file), fix_map)
        return fix_map

    @property
    def dir(self):
        return os.environ["SALICON_DATA_DIR"]

    def prepare_samples(self):
        samples = []
        for file in glob.iglob(self.dir+os.sep+'images'+os.sep+self.phase+os.sep+self.file_stem + '*.jpg'):
            samples.append(int(os.path.basename(file)[-12:-4]))
        return sorted(samples)

    def __len__(self):
        return len(self.samples)

    def preprocess(self, img, data='img'):
        transformations = [
            transforms.ToPILImage(),
        ]
        if data == 'img':
            transformations.append(transforms.Resize(
                self.out_size, interpolation=PIL.Image.LANCZOS))
        transformations.append(transforms.ToTensor())
        if data == 'img' and 'rgb_mean' in self.preproc_cfg:
            transformations.append(
                transforms.Normalize(
                    self.preproc_cfg['rgb_mean'], self.preproc_cfg['rgb_std']))
        elif data == 'sal':
            transformations.append(transforms.Lambda(utils.normalize_tensor))
        elif data == 'fix':
            transformations.append(
                transforms.Lambda(lambda fix: torch.gt(fix, 0.5)))

        processing = transforms.Compose(transformations)
        tensor = processing(img)
        return tensor

    def get_data(self, img_nr):
        img = self.get_img(img_nr)
        img=self.transforms(image=img)

        # img = self.preprocess(img, data='img')
        if self.phase == 'test':
            return [1], img, self.target_size

        sal = self.get_map(img_nr)
        # sal = self.preprocess(sal, data='sal')
        fix,res = self.get_raw_fixations(img_nr)
        fix = self.process_raw_fixations(fix, res)
        fix=preprocess_fixmaps(fix, 192,256)
        fix=fix>125
        sal = cv2.resize(sal,(256,192))
        sal,fix=torch.tensor(sal,dtype=torch.float32),torch.tensor(fix)


        return img, sal, fix, self.target_size

    def __getitem__(self, item):
        img_nr = self.samples[item]
        return self.get_data(img_nr)





# The DataLoader for our specific video datataset with extracted frames
class DHF1K_frames(data.Dataset):

  def __init__(self, phase,split, clip_length, number_of_videos, starting_video, root_path, load_gt, resolution=None, val_perc = 0.01):
        self.phase=phase

        self.starting_video = starting_video
        self.cl = clip_length
        self.frames_path = os.path.join(root_path) # in our case it's salgan saliency maps
        self.load_gt = load_gt
        if load_gt:
          self.gt_path = os.path.join(root_path)#ground truth
        self.ImageNet_mean = [103.939, 116.779, 123.68]
        self.resolution = resolution
        self.transforms = get_train_transforms(self.resolution[0],self.resolution[1])
        # A list to keep all video lists of salgan predictions, which will be our dataset.
        self.video_list = []

        # A list to keep all the dictionaries of ground truth - saliency map pairings for each video
        self.gts_list = []

        start = datetime.datetime.now().replace(microsecond=0) # Gives accurate human readable time, rounded down not to include too many decimals
        for i in range(starting_video, number_of_videos+1): #700 videos in DHF1K

            # The way the folder structure is organized allows to simply iterate over the range of the number of total videos.
            frame_files = os.listdir(os.path.join(self.frames_path,str(i).rjust(4,'0'),'images'))
            #print("for video {} the frames are {}".format(i, len(frame_files))) # This is correct

            # Now to sort based on their file number. The "key" parameter in sorted is a function based on which the sorting will happen (I use split to exclude the jpg/png from the).
            frame_files_sorted = sorted(frame_files, key = lambda x: int(x.split(".")[0]) )
            # a list of lists
            self.video_list.append(frame_files_sorted)

            if load_gt:
              gt_files = os.listdir(os.path.join(self.gt_path,str(i).rjust(4,'0'),'maps'))
              gt_files_sorted = sorted(gt_files, key = lambda x: int(x.split(".")[0]) )
              pack = zip(gt_files_sorted, frame_files_sorted)

              # Make dictionary where keys are the saliency maps and values are the ground truths
              gt_frame_pairings = {}
              for gt, frame in pack:
                  gt_frame_pairings[frame] = gt

              self.gts_list.append(gt_frame_pairings)

            if i%50==0:
              print("Video {} finished.".format(i))
              print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))

        # pack a list of data with the corresponding list of ground truths
        # Split the dataset to validation and training
        limit = int(round(val_perc*len(self.video_list)))
        if split == "validation":
          self.first_video_no = len(self.video_list) - limit+1
          self.video_list = self.video_list[-limit:]
          self.gts_list = self.gts_list[-limit:]
           #This needs to be specified to find the correct directory in our case. It will be different for each split since these directories signify videos.
        elif split == "train":
          self.video_list = self.video_list[:-limit]
          self.gts_list = self.gts_list[:-limit]
          self.first_video_no = starting_video
        elif split == None:
          self.first_video_no = starting_video


        # if split == "validation":
        #   self.video_list = self.video_list[:limit]
        #   self.gts_list = self.gts_list[:limit]
        #   self.first_video_no = starting_video #This needs to be specified to find the correct directory in our case. It will be different for each split since these directories signify videos.
        # elif split == "train":
        #   self.video_list = self.video_list[limit:]
        #   self.gts_list = self.gts_list[limit:]
        #   self.first_video_no = limit+starting_video
        # elif split == None:
        #   self.first_video_no = starting_video




  def __len__(self):
        'Denotes the total number of samples'
        return len(self.video_list)

  def __getitem__(self, video_index):

        'Generates one sample of data'
        # Select sample video (frame list), in our case saliency map list
        frames = self.video_list[video_index]
        if self.load_gt:
          gts = self.gts_list[video_index]

        # Due to the split in train and validation we need to add this number to the video_index to find the correct video (to match the files in the path with the video list the training part uses)
        true_index = self.first_video_no + video_index #this matches the correct video number

        data = []
        gt = []
        gt_fix=[]
        packed = []

        if self.phase=='train':
            v = np.random.random()
            start_idx = np.random.randint(0, len(frames)-self.cl)
            for i, frame in enumerate(frames[start_idx:start_idx+self.cl]):

              # Load and preprocess frames
              path_to_frame = os.path.join(self.frames_path, str(true_index).rjust(4,'0'),'images', frame)

              X = cv2.imread(path_to_frame,cv2.IMREAD_COLOR)
              X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB).astype(np.float32)
              if v<0.5:
                  X = X[:, ::-1, ...]
              if self.resolution!=None:
                # X = cv2.resize(X, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
                X=self.transforms(image=X)['image']
              # X = X.astype(np.float32)
              # X -= self.ImageNet_mean
              #X = (X-np.min(X))/(np.max(X)-np.min(X))
              # X = torch.FloatTensor(X)
              # X = X.permute(2,0,1) # swap channel dimensions
              data.append(X.unsqueeze(0))
              # Load and preprocess ground truth (saliency maps)
              if self.load_gt:

                path_to_gt = os.path.join(self.gt_path, str(true_index).rjust(4,'0'),'maps', gts[frame])
                path_to_gt_fix=os.path.join(self.gt_path, str(true_index).rjust(4,'0'),'fixation', gts[frame])

                map = cv2.imread(str(path_to_gt), cv2.IMREAD_GRAYSCALE)
                map = (map - np.min(map)) / (np.max(map) - np.min(map))
                if v < 0.5:
                    map = map[:, ::-1]

                if self.resolution!=None:
                  map = cv2.resize(map, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
                map = torch.FloatTensor(map)

                fix_map = cv2.imread(str(path_to_gt_fix), cv2.IMREAD_GRAYSCALE)
                if v < 0.5:
                    fix_map= fix_map[:, ::-1]
                fix_map = preprocess_fixmaps(fix_map, self.resolution[0],self.resolution[1])
                fix_map=fix_map>125
                fix_map = torch.tensor(fix_map)

                gt.append(map.unsqueeze(0))
                gt_fix.append(fix_map.unsqueeze(0))

              if (i+1)%self.cl == 0 or i == (len(frames)-1):

                data_tensor = torch.cat(data,0)
                data = []
                if self.load_gt:
                  gt_tensor = torch.cat(gt,0)
                  gt_fix_tensor=torch.cat(gt_fix,0)
                  gt = []
                  gt_fix=[]
                  packed.append((data_tensor,gt_tensor,gt_fix_tensor)) # pack a list of data with the corresponding list of ground truths
                else:
                  packed.append((data_tensor, "_"))

        else:
            for i, frame in enumerate(frames):

                # Load and preprocess frames
                path_to_frame = os.path.join(self.frames_path, str(true_index).rjust(4, '0'), 'images', frame)

                X = cv2.imread(path_to_frame, cv2.IMREAD_COLOR)
                X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB).astype(np.float32)
                if self.resolution != None:
                     X = self.transforms(image=X)['image']

            #salema
                # X = cv2.imread(path_to_frame)
                # if self.resolution != None:
                #     #salema
                #     X = cv2.resize(X, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
                # X = X.astype(np.float32)
                # X -= self.ImageNet_mean
                # X = torch.FloatTensor(X)
                # X = X.permute(2, 0, 1)  # swap channel dimensions


                # X = (X-np.min(X))/(np.max(X)-np.min(X))
                # X = torch.FloatTensor(X)


                data.append(X.unsqueeze(0))
                # Load and preprocess ground truth (saliency maps)
                if self.load_gt:

                    path_to_gt = os.path.join(self.gt_path, str(true_index).rjust(4, '0'), 'maps', gts[frame])
                    path_to_gt_fix = os.path.join(self.gt_path, str(true_index).rjust(4, '0'), 'fixation', gts[frame])

                    map = cv2.imread(str(path_to_gt), cv2.IMREAD_GRAYSCALE)
                    map = (map - np.min(map)) / (np.max(map) - np.min(map))
                    if self.resolution != None:
                        map = cv2.resize(map, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
                    map = torch.FloatTensor(map)

                    fix_map = cv2.imread(str(path_to_gt_fix), cv2.IMREAD_GRAYSCALE)
                    fix_map = preprocess_fixmaps(fix_map, self.resolution[0],self.resolution[1])
                    fix_map = fix_map > 125
                    fix_map = torch.tensor(fix_map)

                    gt.append(map.unsqueeze(0))
                    gt_fix.append(fix_map.unsqueeze(0))

                if (i + 1) % self.cl == 0 or i == (len(frames) - 1):

                    data_tensor = torch.cat(data, 0)
                    data = []
                    if self.load_gt:
                        gt_tensor = torch.cat(gt, 0)
                        gt_fix_tensor = torch.cat(gt_fix, 0)
                        gt = []
                        gt_fix = []
                        packed.append((data_tensor, gt_tensor,
                                       gt_fix_tensor))  # pack a list of data with the corresponding list of ground truths
                    else:
                        packed.append((data_tensor, "_"))

        return packed

# The DataLoader for our specific video datataset with extracted frames
class Hollywood_frames(data.Dataset):

  def __init__(self, clip_length, resolution=None, root_path = "~/work/Hollywood-2/testing/", load_gt = False):
        """
        Frames should be under a folder "images" and ground truths under folder named "maps"
        """

        self.cl = clip_length
        self.root_path = root_path # in our case it's salgan saliency maps
        self.ImageNet_mean = [103.939, 116.779, 123.68]
        self.resolution = resolution
        self.load_gt = load_gt
        # A list to keep all video lists of salgan predictions, which will be our dataset.
        self.video_list = []

        # A list to keep all the dictionaries of ground truth - saliency map pairings for each video
        self.gts_list = []

        # A list to match an index to the video name
        self.video_name_list = []

        sample_list = os.listdir(root_path)
        start = datetime.datetime.now().replace(microsecond=0) # Gives accurate human readable time, rounded down not to include too many decimals
        count = 0
        for i in sample_list:

            frame_files = [os.path.join(self.root_path, str(i), "images", file) for file in  os.listdir(os.path.join(self.root_path, str(i), "images"))]
            #print("for video {} the frames are {}".format(i, len(frame_files))) # This is correct

            # Now to sort based on their file number. The "key" parameter in sorted is a function based on which the sorting will happen (I use split to exclude the jpg/png from the).
            frame_files_sorted = sorted(frame_files)
            # a list of lists
            self.video_list.append(frame_files_sorted)
            self.video_name_list.append(i)

            count += 1

            if count%50==0:
              print("Video {} (Number {}) finished.".format(i, count))
              print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))


  def video_names(self):
      return self.video_name_list

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.video_list)

  def __getitem__(self, video_index):

        'Generates one sample of data'
        # Select sample video (frame list), in our case saliency map list
        frames = self.video_list[video_index]

        data = []
        gt = []
        packed = []
        #print("Frame: {}".format(frames[0]))
        for i, path_to_frame in enumerate(frames):

          X = cv2.imread(path_to_frame)
          if self.resolution!=None:
            X = cv2.resize(X, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
          X = X.astype(np.float32)
          X -= self.ImageNet_mean
          #X = (X-np.min(X))/(np.max(X)-np.min(X))
          X = torch.FloatTensor(X)
          X = X.permute(2,0,1) # swap channel dimensions

          data.append(X.unsqueeze(0))
          # Load and preprocess ground truth (saliency maps)
          if self.load_gt:
            path_to_gt = path_to_frame.replace("images", "maps")
            y = cv2.imread(path_to_gt, 0) # Load as grayscale
            if self.resolution!=None:
              y = cv2.resize(y, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
            y = (y-np.min(y))/(np.max(y)-np.min(y))
            y = torch.FloatTensor(y)

            gt.append(y.unsqueeze(0))

            """
            print("frame: {}".format(path_to_frame))
            print("gtruth: {}".format(path_to_gt))
            exit()
            """

          if (i+1)%self.cl == 0 or i == (len(frames)-1):

            data_tensor = torch.cat(data,0)
            data = []
            if self.load_gt:
              gt_tensor = torch.cat(gt,0)
              gt = []
              packed.append((data_tensor,gt_tensor)) # pack a list of data with the corresponding list of ground truths
            else:
              packed.append((data_tensor, "_"))


        return packed



# The DataLoader for our specific video datataset with extracted frames
class DAVIS_frames(data.Dataset):

  def __init__(self, clip_length, resolution=None, root_path = "~/projects/segmentation/davis2017/JPEGImages/480p", load_gt = False):
        """
        Frames should be under a folder "images" and ground truths under folder named "maps"
        """

        self.cl = clip_length
        self.root_path = root_path # in our case it's salgan saliency maps
        self.ImageNet_mean = [103.939, 116.779, 123.68]
        self.resolution = resolution
        self.load_gt = load_gt
        # A list to keep all video lists of salgan predictions, which will be our dataset.
        self.video_list = []

        # A list to keep all the dictionaries of ground truth - saliency map pairings for each video
        self.gts_list = []

        # A list to match an index to the video name
        self.video_name_list = []

        sample_list = os.listdir(root_path)
        start = datetime.datetime.now().replace(microsecond=0) # Gives accurate human readable time, rounded down not to include too many decimals
        count = 0
        for i in sample_list:

            frame_files = [os.path.join(self.root_path, str(i), file) for file in os.listdir(os.path.join(self.root_path, str(i)))]
            #print("for video {} the frames are {}".format(i, len(frame_files))) # This is correct

            # Now to sort based on their file number. The "key" parameter in sorted is a function based on which the sorting will happen (I use split to exclude the jpg/png from the).
            frame_files_sorted = sorted(frame_files)
            # a list of lists
            self.video_list.append(frame_files_sorted)
            self.video_name_list.append(i)

            count += 1

            if count%50==0:
              print("Video {} (Number {}) finished.".format(i, count))
              print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))


  def video_names(self):
      return self.video_name_list

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.video_list)

  def __getitem__(self, video_index):

        'Generates one sample of data'
        # Select sample video (frame list), in our case saliency map list
        frames = self.video_list[video_index]

        data = []
        gt = []
        packed = []
        #print("Frame: {}".format(frames[0]))
        for i, path_to_frame in enumerate(frames):

          X = cv2.imread(path_to_frame)
          if self.resolution!=None:
            X = cv2.resize(X, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
          X = X.astype(np.float32)
          X -= self.ImageNet_mean
          #X = (X-np.min(X))/(np.max(X)-np.min(X))
          X = torch.FloatTensor(X)
          X = X.permute(2,0,1) # swap channel dimensions

          data.append(X.unsqueeze(0))
          # Load and preprocess ground truth (saliency maps)
          if self.load_gt:
            path_to_gt = path_to_frame.replace("images", "maps")
            y = cv2.imread(path_to_gt, 0) # Load as grayscale
            if self.resolution!=None:
              y = cv2.resize(y, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
            y = (y-np.min(y))/(np.max(y)-np.min(y))
            y = torch.FloatTensor(y)

            gt.append(y.unsqueeze(0))

            """
            print("frame: {}".format(path_to_frame))
            print("gtruth: {}".format(path_to_gt))
            exit()
            """

          if (i+1)%self.cl == 0 or i == (len(frames)-1):

            data_tensor = torch.cat(data,0)
            data = []
            if self.load_gt:
              gt_tensor = torch.cat(gt,0)
              gt = []
              packed.append((data_tensor,gt_tensor)) # pack a list of data with the corresponding list of ground truths
            else:
              packed.append((data_tensor, "_"))


        return packed




