import cv2
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import os
import glob
import logging
import sklearn.preprocessing
from tqdm import tqdm
import h5py

OPENBLAS_NUM_THREADS = 1

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.tiff', '.TIF', '.TIFF'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def black_pixels(img):
    return np.sum(img == 0)
def white_pixels(img):
    return np.sum(img==255)


def calc_features(input_img, arguments):
    img = cv2.imread(input_img)
    #print(f'image name:{os.path.basename(input_img)}')
    if arguments.scale != -1:
        scale = arguments.scale
        height = img.shape[0] * scale
        width = img.shape[1] * scale
        new_size_mask = (int(width), int(height))
        img = cv2.resize(img, new_size_mask, interpolation=cv2.INTER_CUBIC)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img_gray.copy()

    sift = cv2.SIFT_create(sigma=args.sigma)        # sigma is responsible to generate more / less  key points
    kp = sift.detect(img_gray, None)                # if sigma high then less kp detected if not then otherwise
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    def filter_keypoint(point):
        # if 3 < point.size < 10:
        #     return point
        return point


    new_kp = []
    uniq_kp = []
    black_pixel_filtered = 0
    white_pixel_filtered = 0
    kept_patches = 0
    num_pixel = arguments.win_size ** 2
    for p in tqdm(kp, 'New Keypoints generation'):
        point = (int(p.pt[0]), int(p.pt[1]))

        roi = get_image_patch(img, point[0], point[1], arguments)
        
        if roi is None:
            continue

        if arguments.centered:
            if img_bin[point[1], point[0]] != 0:
                continue

        if arguments.black_pixel_thresh != -1:
            if black_pixels(roi) > (arguments.black_pixel_thresh * num_pixel):
                black_pixel_filtered+=1
                continue 

        if arguments.white_pixel_thresh != -1:
            if white_pixels(roi) > (arguments.white_pixel_thresh * num_pixel):
                white_pixel_filtered+=1
                continue


        if not point in uniq_kp and filter_keypoint(p):
            new_kp.append(p)
            uniq_kp.append(point)
            kept_patches += 1
            
    print(f'Image: {os.path.basename(input_img)}')
    print(f'Black pixel filtered patches: {black_pixel_filtered}')
    print(f'White pixel filtered patches: {white_pixel_filtered}')
    print(f'Kept patches: {kept_patches}')
    
    if not new_kp:
        return []
    _, desc = sift.compute(img_gray, new_kp) # this is (new_kp,128) matrix called descriptor
    # descriptors are 128D vector which encodes information about the local gradient directions and magnitudes around that keypoint.
    desc = sklearn.preprocessing.normalize(desc, norm='l1')
    desc = np.sign(desc) * np.sqrt(np.abs(desc))
    desc = sklearn.preprocessing.normalize(desc, norm='l2')

    kp_tupple = [(p, desc[i]) for i, p in enumerate(uniq_kp)]

    return kp_tupple


def get_image_patch(img, px, py, arguments):
    half_win_size = int(arguments.win_size / 2)
    if not (half_win_size < px < img.shape[1] - half_win_size and half_win_size < py < img.shape[0] - half_win_size):
        # if px - half_win_size < 0 or px + half_win_size > img.shape[1] or py - half_win_size < 0 or py + half_win_size > img.shape[0]:
        return None

    roi = img[py - half_win_size:py + half_win_size, px - half_win_size:px + half_win_size]
    assert roi.shape == (half_win_size * 2, half_win_size * 2), 'shape of the roi is not (%d,%d). It is (%d,%d)' % \
                                                                (half_win_size * 2, half_win_size * 2,
                                                                 roi.shape[0], roi.shape[1])
    return roi


def extract_patches(filename, tup, args):
    if len(tup) > args.patches_per_page and args.patches_per_page != -1:
        idx = np.linspace(0, len(tup) - 1, args.patches_per_page, dtype=np.int32)
        tup = [tup[i] for i in idx]

    points, clusters = zip(*tup)

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args.scale != -1:
        scale = args.scale
        height = img.shape[0] * scale
        width = img.shape[1] * scale
        new_size_mask = (int(width), int(height))
        img = cv2.resize(img, new_size_mask, interpolation=cv2.INTER_CUBIC)

    count = 0
    #cluster_h5_files = {}
    
    for p, c in zip(points, clusters):
        roi = get_image_patch(img, p[0], p[1], args) # These are final patch extraction after transformation and filter
        if roi is None:
            continue
        
        # structure
        # - cluster_0
        #       -- patch image 0 : 32x32x1
        #       -- patch image 1 : 32x32x1
        # - cluster_1      
        
        h5_file_path = os.path.join(args.out_dir[0],f'cluster_{c}.h5')
        
        with h5py.File(h5_file_path,'a') as h5f:
            img_filename = f'{c}_{os.path.splitext(os.path.basename(filename))[0]}_{count}'
            
            if img_filename in h5f:
                logging.warning(f'image file {img_filename} already exist in {h5_file_path}, skipping...')
                continue
            h5f.create_dataset(img_filename, data=roi, compression="gzip")            
        count = count + 1
    


if __name__ == "__main__":
    import argparse
    import pickle

    np.random.seed(42)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')

    parser = argparse.ArgumentParser(description="extract patches from images")
    

    parser.add_argument('--in_dir', metavar='in_dir', dest='in_dir', type=str, nargs=1,
                        help='input directory', required=True)
    parser.add_argument('--out_dir', metavar='out_dir', dest='out_dir', type=str, nargs=1,
                        help='output directory', required=True)
    parser.add_argument('--win_size', metavar='win_size', dest='win_size', type=int, nargs='?',
                        help='size of the patch',
                        default=32)
    parser.add_argument('--num_of_clusters', metavar='num_of_clusters', dest='number_of_clusters', type=int, nargs='?',
                        help='number of clusters',
                        default=-1)
    parser.add_argument('--patches_per_page', metavar='patches_per_page', dest='patches_per_page', type=int, nargs='?',
                        help='maximal number of patches per page (-1 for no limit)',
                        default=-1)
    parser.add_argument('--scale', type=float, help='scale images up or down',
                        default=-1)
    parser.add_argument('--sigma', type=float, help='blur factor for SIFT',
                        default=1.6)
    parser.add_argument('--black_pixel_thresh', type=float, help='if more black_pixel_thresh percent of the pixels are black -> discard',
                        default=0.5)
    parser.add_argument('--white_pixel_thresh', type=float, help='if more than white_pixel_thresh percent of the pixels are white -> discard',
                        default=0.5)
    parser.add_argument('--centered', type=bool, help='filter patches whose keypoints are not located on handwriting, only for binarized datasets',
                    default=True)

    args = parser.parse_args()


    assert os.path.exists(args.in_dir[0]), 'in_dir {} does not exist'.format(args.in_dir[0])


    if not os.path.exists(args.out_dir[0]):
        logging.info('creating directory %s' % args.out_dir[0])
        os.mkdir(args.out_dir[0])

    assert len(os.listdir(args.out_dir[0])) == 0, 'out_dir is not empty'

    assert args.win_size % 2 == 0, 'win_size must be even'

    num_cores = int(multiprocessing.cpu_count() / 2)
    num_cores = 10
    path_to_centers = ''

    files = [f for f in glob.glob(args.in_dir[0] + '/**/*.*', recursive=True) if os.path.isfile(f) and is_image_file(f)]

    assert len(files) > 0, 'no images found'
    logging.info('Found {} images'.format(len(files)))

    logging.info('calculating features for images in %s (number of cores:%d)' % (args.in_dir, num_cores))
    results = []
    results = Parallel(n_jobs=num_cores, verbose=9)(delayed(calc_features)(f, args) for f in files)

    logging.info('collecting descriptors')
    desc_list = []
    kp_list = []
    fn_list = []
    for r, f in tqdm(zip(results, files)):
        if len(r) == 0:
            logging.warning('no keypoints found in file {} '.format(f))
        for kp, desc in r:
            kp_list.append(kp)      # keypoints
            desc_list.append(desc)  # 128D vector
            fn_list.append(f)       # original image
    results = None

    logging.info('calculating pca (number of patches: {})'.format(len(desc_list)))
    from sklearn.decomposition import PCA

    # reducing descriptor dimensions from 128D to 32 (patches,128D) per image
    pca = PCA(32, whiten=True) # whiten means : Normalizes the output of PCA so that each component has unit variance.
    desc = pca.fit_transform(np.array(desc_list))
    desc_list = None

    logging.info('calculating new centers (shape: {})'.format(desc.shape)) # shape=> (number of patches, 32)

    logging.info("starting KMeans (centers: {})".format(args.number_of_clusters)) #=> 5000
    # km = KMeans(n_clusters=number_of_clusters, random_state=0, n_jobs=-1, verbose=0).fit(desc)
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import DBSCAN
    
    if desc.shape[0] > 500000:
        idx = np.linspace(0, desc.shape[0]-1, 500000, dtype=np.int32)
        sampled_desc = desc[idx]
    else:
        sampled_desc = desc

    
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(desc)
    distances, indices = neighbors_fit.kneighbors(desc)
    
    # this will calculate centroid but it won't assign samples to centroid/clusters as compute_labels=False
    # this is only instance creation
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=args.number_of_clusters, compute_labels=False, 
                                            batch_size=10000 if args.number_of_clusters <= 1000 else 50000)
    

    # total patches for all images
    if desc.shape[0] > 500000:
       idx = np.linspace(0, desc.shape[0]-1, 500000, dtype=np.int32)
       kmeans.fit(desc[idx])
    else:
       kmeans.fit(desc)
    
    center_path = os.path.join(args.out_dir[0], 'centers.pkl')
    logging.info('saving centers to %s' % center_path)
       
    patches_files = {}      # This will store final patches after filtering based on ro(p) ratio > 0.9
    feature_count = 0
    batch_size = 50000
    for batch_start in tqdm(range(0, desc.shape[0], batch_size), 'Transforming and filtering'):
        def batch(d):
            return d[batch_start:batch_start+batch_size]
        dist = kmeans.transform(batch(desc))            #(number of samples,number of clusters)
        prediction = kmeans.predict(batch(desc))
        #print(f'shape of dist and predictions:{dist.shape},{prediction.shape}')
        dist = np.sort(dist)
        ratio = dist[:, 0] / dist[:, 1]     # 0 -> smallest distance from any cluster, 1 -> second smallest distance

        # This is filtering step which will filter out few or more KP and patches
        for p, f, c, r in zip(batch(kp_list), batch(fn_list), prediction, ratio):
            if r <= 0.9:
                feature_count += 1
                if f in patches_files:
                    patches_files[f].append((p, c))
                else:
                    patches_files[f] = [(p, c)]

    logging.info('Final patches :copying %i (all patches per page) image patches to %s ' % (feature_count, args.out_dir[0]))

    for filename, tup in tqdm(patches_files.items(),'Image patches generation'):
        extract_patches(filename, tup, args)

    config_out_path = os.path.join(args.out_dir[0], 'db-creation-parameters.json')
    logging.info(f'writing config parameters to {config_out_path}')
    with open(config_out_path, 'w') as f:
        import json

        json.dump(vars(args), f)
    
    logging.info('done')
